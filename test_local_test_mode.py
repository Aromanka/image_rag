"""Model-free tests for the local display WebSocket mode."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

import image_server
from utils.local_test_channel import LocalTestHub
from utils.local_test_data import load_display_samples
from utils.local_test_display import (
    JsonlWriter,
    infer_dataset_from_annotations,
    parse_args as local_test_display_parse_args,
    read_recorded_event_ids,
    websocket_uri,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.messages: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        self.messages.append(payload)


class LocalTestChannelTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_and_replay_are_ordered(self) -> None:
        hub = LocalTestHub(enabled=True, history_size=4)
        first = FakeWebSocket()
        await hub.connect(first)
        published = await hub.publish(
            {"type": "inference.completed", "event_id": "event-1"}
        )

        self.assertTrue(first.accepted)
        self.assertEqual(first.messages[0]["type"], "local_test.ready")
        self.assertEqual(first.messages[1]["sequence"], 1)
        self.assertEqual(published["server_instance_id"], hub.instance_id)

        await hub.disconnect(first)
        second = FakeWebSocket()
        await hub.connect(
            second,
            after_sequence=0,
            client_instance_id=hub.instance_id,
        )
        self.assertEqual(second.messages[0]["replayed_count"], 1)
        self.assertEqual(second.messages[1]["event_id"], "event-1")

    async def test_history_limit(self) -> None:
        hub = LocalTestHub(enabled=True, history_size=2)
        for index in range(3):
            await hub.publish(
                {"type": "inference.completed", "event_id": f"event-{index}"}
            )
        client = FakeWebSocket()
        await hub.connect(
            client,
            after_sequence=0,
            client_instance_id=hub.instance_id,
        )
        self.assertTrue(client.messages[0]["replay_limited"])
        self.assertEqual(
            [message["event_id"] for message in client.messages[1:]],
            ["event-1", "event-2"],
        )

    async def test_fresh_session_skips_old_events_but_new_instance_replays(self) -> None:
        hub = LocalTestHub(enabled=True, history_size=4)
        await hub.publish(
            {"type": "inference.completed", "event_id": "old-event"}
        )

        fresh_client = FakeWebSocket()
        await hub.connect(fresh_client)
        self.assertEqual(fresh_client.messages[0]["replayed_count"], 0)
        self.assertEqual(len(fresh_client.messages), 1)
        await hub.disconnect(fresh_client)

        restarted_client = FakeWebSocket()
        await hub.connect(
            restarted_client,
            after_sequence=99,
            client_instance_id="previous-server-instance",
        )
        self.assertEqual(restarted_client.messages[0]["replayed_count"], 1)
        self.assertEqual(restarted_client.messages[1]["event_id"], "old-event")


class LocalTestDatasetTests(unittest.TestCase):
    def _image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(path)

    def test_labsafety_jsonl_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "test" / "lab.png"
            self._image(image_path)
            annotations = root / "annotations.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "image_id": "lab-1",
                        "image": "images/test/lab.png",
                        "split": "test",
                        "safety_label": "hazardous",
                        "hazards": ["spill"],
                        "description": "A spill is visible.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            samples, missing = load_display_samples(
                dataset="labsafety_gen",
                annotations_path=annotations,
                image_root=None,
                split="test",
            )

            self.assertEqual(missing, 0)
            self.assertEqual(samples[0].sample_id, "lab-1")
            self.assertEqual(samples[0].ground_truth["safety_label"], "hazardous")
            self.assertEqual(samples[0].image_path, image_path)

    def test_inspecsafe_flat_image_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flat_root = root / "flat"
            image_path = flat_root / "test__site-Level01-0001__frame.jpg"
            self._image(image_path)
            annotations = root / "pipeline.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "image": "images/test__site-Level01-0001__frame.jpg",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "hazards": ["smoke"],
                                            "overall_safety_level": "Level I",
                                        }
                                    ),
                                }
                            ],
                            "metadata": {"split": "test"},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            samples, missing = load_display_samples(
                dataset="inspecsafe_safety_level",
                annotations_path=annotations,
                image_root=flat_root,
                split="test",
            )

            self.assertEqual(missing, 0)
            self.assertEqual(samples[0].image_path, image_path)
            self.assertEqual(
                samples[0].ground_truth["overall_safety_level"],
                "Level I",
            )

    def test_missing_images_can_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            annotations = Path(temp_dir) / "annotations.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "image_id": "missing",
                        "image": "images/test/missing.png",
                        "split": "test",
                        "safety_label": "non-hazardous",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples, missing = load_display_samples(
                dataset="labsafety_gen",
                annotations_path=annotations,
                image_root=None,
                split="test",
            )
            self.assertEqual(samples, [])
            self.assertEqual(missing, 1)


class LocalTestProtocolTests(unittest.TestCase):
    def test_display_cli_defaults_to_local_ssh_forward_port(self) -> None:
        with patch("sys.argv", ["local_test_display.py"]):
            args = local_test_display_parse_args()
        self.assertEqual(
            args.server,
            "ws://127.0.0.1:18000/local-test/ws",
        )

    def test_dataset_is_inferred_from_portable_batch_annotations(self) -> None:
        self.assertEqual(
            infer_dataset_from_annotations(
                Path("data/local_test_batch/inspecsafe_safety_level/annotations.json")
            ),
            "inspecsafe_safety_level",
        )
        self.assertEqual(
            infer_dataset_from_annotations(
                Path("data/local_test_batch/labsafety_gen/annotations.jsonl")
            ),
            "labsafety_gen",
        )

    def test_completion_event_contains_query_identity_and_full_result(self) -> None:
        result = {"status": "success", "safe": "safe", "response": "safe"}
        event = image_server._build_local_test_event(
            payload=b"query bytes",
            content_type="image/jpeg",
            dataset="labsafety_gen",
            sample_id="lab-1",
            response_payload=result,
        )
        self.assertEqual(event["type"], "inference.completed")
        self.assertEqual(event["query"]["size_bytes"], 11)
        self.assertEqual(event["query"]["sample_id"], "lab-1")
        self.assertIs(event["result"], result)

    def test_client_uri_preserves_query_and_resume_state(self) -> None:
        uri = websocket_uri(
            "wss://example.test/ws?client=display",
            after_sequence=7,
            server_instance_id="server-1",
        )
        self.assertIn("client=display", uri)
        self.assertIn("after_sequence=7", uri)
        self.assertIn("server_instance_id=server-1", uri)

    def test_jsonl_writer_flushes_complete_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results" / "trials.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps({"event_id": "stale-event"}) + "\n",
                encoding="utf-8",
            )
            writer = JsonlWriter(output)
            writer.start()
            writer.submit(
                {
                    "event_id": "event-1",
                    "sample": {"ground_truth": {"safety_label": "hazardous"}},
                    "server_result": {"safe": "unsafe"},
                }
            )
            writer.close()

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["server_result"]["safe"], "unsafe")
            self.assertEqual(read_recorded_event_ids(output), {"event-1"})

    def test_infer_completion_reaches_websocket_client(self) -> None:
        app = image_server.create_app(
            preload=False,
            local_test_mode=True,
            local_test_dataset="labsafety_gen",
        )
        image_buffer = BytesIO()
        Image.new("RGB", (8, 8), "green").save(image_buffer, format="JPEG")

        client = TestClient(app)
        try:
            with client.websocket_connect(
                "/local-test/ws?after_sequence=-1"
            ) as websocket:
                ready = websocket.receive_json()
                self.assertEqual(ready["type"], "local_test.ready")

                with patch.object(
                    image_server,
                    "_run_inference",
                    return_value=(
                        "unsafe Smoke is visible.",
                        {"label": "unsafe", "annotation": "Smoke is visible."},
                    ),
                ):
                    response = client.post(
                        "/infer?mode=latency&local_test_sample_id=lab-1",
                        content=image_buffer.getvalue(),
                        headers={"Content-Type": "image/jpeg"},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.headers["X-Local-Test-Notification"],
                    "scheduled",
                )
                event = websocket.receive_json()
                self.assertEqual(event["type"], "inference.completed")
                self.assertEqual(event["query"]["dataset"], "labsafety_gen")
                self.assertEqual(event["query"]["sample_id"], "lab-1")
                self.assertEqual(event["result"]["safe"], "unsafe")
                self.assertEqual(
                    event["result"]["annotation"],
                    "Smoke is visible.",
                )
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
