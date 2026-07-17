"""Reliable in-memory WebSocket fan-out for image-server local tests."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


LOCAL_TEST_PROTOCOL_VERSION = 1


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LocalTestHub:
    """Track display clients and replay recent inference-completion events."""

    def __init__(
        self,
        *,
        enabled: bool,
        history_size: int = 1024,
        send_timeout_seconds: float = 2.0,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1.")
        if send_timeout_seconds <= 0:
            raise ValueError("send_timeout_seconds must be positive.")

        self.enabled = enabled
        self.history_size = history_size
        self.send_timeout_seconds = send_timeout_seconds
        self.instance_id = str(uuid4())
        self._sequence = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._clients: dict[int, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        return len(self._clients)

    async def connect(
        self,
        websocket: Any,
        *,
        after_sequence: int = -1,
        client_instance_id: str | None = None,
    ) -> None:
        """Accept a client and replay events missed by the same server instance.

        ``after_sequence=-1`` starts a fresh display session at the latest
        event. A reconnect supplies both its last sequence and instance ID. If
        the server restarted, all retained events from the new instance are
        replayed instead of comparing unrelated sequence numbers.
        """
        await websocket.accept()
        async with self._lock:
            if after_sequence == -1 and client_instance_id is None:
                effective_after_sequence = self._sequence
            elif client_instance_id != self.instance_id:
                effective_after_sequence = 0
            else:
                effective_after_sequence = max(after_sequence, 0)
            oldest_sequence = (
                int(self._history[0]["sequence"]) if self._history else None
            )
            replay_limited = (
                oldest_sequence is not None
                and effective_after_sequence < oldest_sequence - 1
            )
            replay = [
                event
                for event in self._history
                if int(event["sequence"]) > effective_after_sequence
            ]
            await self._send_json(
                websocket,
                {
                    "type": "local_test.ready",
                    "protocol_version": LOCAL_TEST_PROTOCOL_VERSION,
                    "server_instance_id": self.instance_id,
                    "connected_at": utc_now_iso(),
                    "latest_sequence": self._sequence,
                    "replay_from_sequence": effective_after_sequence,
                    "replayed_count": len(replay),
                    "replay_limited": replay_limited,
                },
            )
            for event in replay:
                await self._send_json(websocket, event)
            self._clients[id(websocket)] = websocket

    async def disconnect(self, websocket: Any) -> None:
        async with self._lock:
            self._clients.pop(id(websocket), None)

    async def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        """Assign ordering metadata, retain, and broadcast one event."""
        async with self._lock:
            self._sequence += 1
            published = {
                **event,
                "protocol_version": LOCAL_TEST_PROTOCOL_VERSION,
                "server_instance_id": self.instance_id,
                "sequence": self._sequence,
            }
            self._history.append(published)
            clients = list(self._clients.values())

        if not clients:
            return published

        results = await asyncio.gather(
            *(self._try_send(client, published) for client in clients),
            return_exceptions=False,
        )
        dead_clients = [
            client for client, succeeded in zip(clients, results) if not succeeded
        ]
        if dead_clients:
            async with self._lock:
                for client in dead_clients:
                    self._clients.pop(id(client), None)
        return published

    async def _try_send(self, websocket: Any, payload: dict[str, Any]) -> bool:
        try:
            await self._send_json(websocket, payload)
        except Exception:
            return False
        return True

    async def _send_json(self, websocket: Any, payload: dict[str, Any]) -> None:
        await asyncio.wait_for(
            websocket.send_json(payload),
            timeout=self.send_timeout_seconds,
        )
