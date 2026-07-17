"""Fullscreen dataset display driven by image-server WebSocket events."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import random
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageOps, ImageTk  # noqa: E402

from utils.local_test_data import (  # noqa: E402
    DisplaySample,
    SUPPORTED_LOCAL_TEST_DATASETS,
    default_annotations_path,
    load_display_samples,
    normalize_dataset,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def websocket_uri(
    base_uri: str,
    token: str | None,
    *,
    after_sequence: int = -1,
    server_instance_id: str | None = None,
) -> str:
    """Add local-test connection parameters without discarding existing ones."""
    parsed = urlsplit(base_uri)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("--server must use ws:// or wss://.")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["after_sequence"] = str(after_sequence)
    if server_instance_id:
        query["server_instance_id"] = server_instance_id
    if token:
        query["token"] = token
    return urlunsplit(parsed._replace(query=urlencode(query)))


def read_recorded_event_ids(path: Path) -> set[str]:
    """Read event IDs already committed to an existing JSONL output."""
    event_ids: set[str] = set()
    if not path.is_file():
        return event_ids
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = record.get("event_id") if isinstance(record, dict) else None
            if event_id:
                event_ids.add(str(event_id))
    return event_ids


class JsonlWriter:
    """Append records off the UI thread and flush every completed line."""

    def __init__(self, path: Path, *, fsync: bool = False) -> None:
        self.path = path
        self.fsync = fsync
        self.records: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.errors: queue.Queue[Exception] = queue.Queue()
        self.thread = threading.Thread(
            target=self._run,
            name="local-test-jsonl-writer",
            daemon=True,
        )

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Fail before the experiment starts if the target cannot be opened.
        with self.path.open("a", encoding="utf-8"):
            pass
        self.thread.start()

    def submit(self, record: dict[str, Any]) -> None:
        self.records.put(record)

    def close(self) -> None:
        self.records.put(None)
        self.thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as file:
                while True:
                    record = self.records.get()
                    if record is None:
                        break
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file.flush()
                    if self.fsync:
                        os.fsync(file.fileno())
        except Exception as exc:
            self.errors.put(exc)


class EventReceiver:
    """Receive WebSocket events with automatic exponential-backoff reconnects."""

    def __init__(
        self,
        *,
        server_uri: str,
        token: str | None,
        messages: queue.Queue[dict[str, Any]],
        statuses: queue.Queue[str],
    ) -> None:
        self.server_uri = server_uri
        self.token = token
        self.messages = messages
        self.statuses = statuses
        self.stop_event = threading.Event()
        self.last_sequence = -1
        self.server_instance_id: str | None = None
        self.thread = threading.Thread(
            target=self._thread_main,
            name="local-test-websocket",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._receive_forever())
        except Exception as exc:
            self.statuses.put(f"WebSocket receiver stopped: {exc}")

    async def _receive_forever(self) -> None:
        import websockets

        delay = 0.5
        while not self.stop_event.is_set():
            try:
                uri = websocket_uri(
                    self.server_uri,
                    self.token,
                    after_sequence=self.last_sequence,
                    server_instance_id=self.server_instance_id,
                )
                self.statuses.put(f"Connecting to {self.server_uri}")
                async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2 * 1024 * 1024,
                ) as websocket:
                    self.statuses.put("WebSocket connected")
                    delay = 0.5
                    async for raw_message in websocket:
                        if self.stop_event.is_set():
                            return
                        if not isinstance(raw_message, str):
                            continue
                        payload = json.loads(raw_message)
                        if isinstance(payload, dict):
                            if payload.get("type") == "local_test.ready":
                                self.server_instance_id = str(
                                    payload.get("server_instance_id") or ""
                                ) or None
                                self.last_sequence = int(
                                    payload.get("replay_from_sequence", 0)
                                )
                            elif payload.get("type") == "inference.completed":
                                self.last_sequence = max(
                                    self.last_sequence,
                                    int(payload.get("sequence", 0)),
                                )
                            self.messages.put(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.statuses.put(
                    f"WebSocket disconnected ({exc}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)


class LocalTestDisplay:
    """Tk image viewer that advances once per completed inference."""

    POLL_INTERVAL_MS = 8

    def __init__(
        self,
        *,
        samples: list[DisplaySample],
        output_path: Path,
        server_uri: str,
        token: str | None,
        fullscreen: bool,
        loop: bool,
        fsync: bool,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.samples = samples
        self.loop = loop
        self.index = 0
        self.current_displayed_at = ""
        self.current_display_started = 0.0
        self.prepared_next: tuple[int, Image.Image] | None = None
        self.seen_event_ids = read_recorded_event_ids(output_path)
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.statuses: queue.Queue[str] = queue.Queue()
        self.writer = JsonlWriter(output_path, fsync=fsync)
        self.receiver = EventReceiver(
            server_uri=server_uri,
            token=token,
            messages=self.messages,
            statuses=self.statuses,
        )

        self.root = tk.Tk()
        self.root.title("Image_RAG local test display")
        self.root.configure(background="black")
        self.root.attributes("-fullscreen", fullscreen)
        if not fullscreen:
            self.root.geometry("1280x720")
        self.root.bind("<Escape>", lambda _: self.close())
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.label = tk.Label(
            self.root,
            background="black",
            foreground="white",
            borderwidth=0,
            highlightthickness=0,
        )
        self.label.pack(fill="both", expand=True)
        self.photo: ImageTk.PhotoImage | None = None
        self.closed = False
        self.failed = False

    def run(self) -> None:
        self.root.update_idletasks()
        self._show_index(0)
        self.writer.start()
        self.receiver.start()
        self.root.after(self.POLL_INTERVAL_MS, self._poll)
        print(f"Displaying {len(self.samples)} samples. Esc exits; F11 toggles fullscreen.")
        print(f"Writing results to: {self.writer.path}")
        self.root.mainloop()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.receiver.stop()
        self.writer.close()
        self.root.destroy()

    def _toggle_fullscreen(self, _: object) -> None:
        enabled = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not enabled)

    def _poll(self) -> None:
        if self.closed:
            return
        while True:
            try:
                status = self.statuses.get_nowait()
            except queue.Empty:
                break
            print(status, flush=True)
        if not self.writer.errors.empty():
            error = self.writer.errors.get_nowait()
            self._fail(f"JSONL writer failed: {error}")
            return

        while True:
            try:
                event = self.messages.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
            if self.failed:
                return
        self.root.after(self.POLL_INTERVAL_MS, self._poll)

    def _fail(self, message: str) -> None:
        """Stop accepting results rather than advancing without a durable log."""
        if self.failed:
            return
        self.failed = True
        self.receiver.stop()
        print(message, file=sys.stderr, flush=True)
        self.photo = None
        self.label.configure(
            image="",
            text=message + "\n\nPress Esc to exit.",
            font=("Arial", 24),
            wraplength=max(self.root.winfo_width() - 80, 400),
        )

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "local_test.ready":
            if event.get("replay_limited"):
                print(
                    "Warning: server replay history was exhausted; some events may be missing.",
                    file=sys.stderr,
                    flush=True,
                )
            return
        if event_type != "inference.completed":
            return

        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in self.seen_event_ids:
            return
        self.seen_event_ids.add(event_id)

        sample = self.samples[self.index] if self.index < len(self.samples) else None
        displayed_at = self.current_displayed_at
        display_seconds = (
            time.perf_counter() - self.current_display_started
            if self.current_display_started
            else None
        )
        association = self._association_status(sample, event)

        record: dict[str, Any] = {
            "recorded_at": utc_now_iso(),
            "event_id": event_id,
            "server_instance_id": event.get("server_instance_id"),
            "sequence": event.get("sequence"),
            "association": association,
            "displayed_at": displayed_at,
            "display_duration_seconds": (
                round(display_seconds, 6) if display_seconds is not None else None
            ),
            "sample": sample.as_record() if sample is not None else None,
            "server_completed_at": event.get("completed_at"),
            "server_query": event.get("query"),
            "server_result": event.get("result"),
        }
        # Queueing is memory-only and keeps the UI path independent of disk
        # latency. Do it before rendering so a corrupt next image cannot discard
        # the result for the valid image that was just evaluated.
        self.writer.submit(record)
        try:
            self._advance()
        except OSError as exc:
            self._fail(f"Unable to display the next image: {exc}")

    def _association_status(
        self,
        sample: DisplaySample | None,
        event: dict[str, Any],
    ) -> str:
        if sample is None:
            return "no_current_sample"
        query = event.get("query")
        if not isinstance(query, dict):
            return "sequential"
        expected_dataset = query.get("dataset")
        expected_sample_id = query.get("sample_id")
        if expected_dataset and str(expected_dataset) != sample.dataset:
            return "dataset_mismatch"
        if expected_sample_id and str(expected_sample_id) != sample.sample_id:
            return "sample_id_mismatch"
        return "matched_hint" if expected_dataset or expected_sample_id else "sequential"

    def _advance(self) -> None:
        next_index = self.index + 1
        if next_index >= len(self.samples):
            if self.loop:
                next_index = 0
            else:
                self.index = len(self.samples)
                self.photo = None
                self.label.configure(
                    image="",
                    text="Dataset complete",
                    font=("Arial", 32),
                )
                return
        self._show_index(next_index)

    def _show_index(self, index: int) -> None:
        self.index = index
        if self.prepared_next is not None and self.prepared_next[0] == index:
            prepared = self.prepared_next[1]
        else:
            prepared = self._prepare_image(self.samples[index].image_path)
        self.prepared_next = None
        self.photo = ImageTk.PhotoImage(prepared)
        self.label.configure(image=self.photo, text="")
        self.current_displayed_at = utc_now_iso()
        self.current_display_started = time.perf_counter()
        self.root.after_idle(self._preload_following)

    def _preload_following(self) -> None:
        if self.closed or not self.samples:
            return
        next_index = self.index + 1
        if next_index >= len(self.samples):
            if not self.loop:
                return
            next_index = 0
        try:
            prepared = self._prepare_image(self.samples[next_index].image_path)
        except OSError as exc:
            print(f"Unable to preload image: {exc}", file=sys.stderr, flush=True)
            return
        self.prepared_next = (next_index, prepared)

    def _prepare_image(self, path: Path) -> Image.Image:
        if bool(self.root.attributes("-fullscreen")):
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
        else:
            width = max(self.root.winfo_width(), 1)
            height = max(self.root.winfo_height(), 1)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            return ImageOps.contain(
                image,
                (width, height),
                method=Image.Resampling.LANCZOS,
            ).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display local dataset images and advance after image_server "
            "completes each query."
        )
    )
    parser.add_argument(
        "--server",
        default="ws://127.0.0.1:8000/local-test/ws",
        help="image_server local-test WebSocket URL.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("IMAGE_RAG_LOCAL_TEST_TOKEN"),
        help="Optional shared token (or set IMAGE_RAG_LOCAL_TEST_TOKEN).",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_LOCAL_TEST_DATASETS),
        default="inspecsafe_safety_level",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Dataset JSON/JSONL. A repository default is selected by dataset.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional image root. Supports the original InspecSafe DATA_PATH, "
            "a flat InspecSafe image directory, or a LabSafety dataset root."
        ),
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--strict-images", action="store_true")
    parser.add_argument("--fsync", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Append-only result JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise SystemExit("--offset cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    dataset = normalize_dataset(args.dataset)
    annotations = args.annotations or default_annotations_path(dataset, PROJECT_ROOT)
    samples, missing = load_display_samples(
        dataset=dataset,
        annotations_path=annotations,
        image_root=args.image_root,
        split=args.split,
        skip_missing=not args.strict_images,
    )
    if missing:
        print(f"Skipped {missing} samples whose local image was not found.")
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    samples = samples[args.offset :]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(
            "No displayable samples remain. Check --annotations, --image-root, "
            "--split, --offset, and --limit."
        )

    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "save" / f"local_test_{dataset}_{timestamp}.jsonl"

    display = LocalTestDisplay(
        samples=samples,
        output_path=output,
        server_uri=args.server,
        token=args.token,
        fullscreen=not args.windowed,
        loop=args.loop,
        fsync=args.fsync,
    )
    display.run()


if __name__ == "__main__":
    main()
