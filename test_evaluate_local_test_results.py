"""Tests for local display JSONL accuracy metrics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from utils.evaluate_local_test_results import evaluate_jsonl, evaluate_records


def _record(
    *,
    event_id: str,
    dataset: str,
    ground_truth: dict,
    prediction: str | None,
    response: str,
    mode: str = "accuracy",
) -> dict:
    return {
        "event_id": event_id,
        "association": "sequential",
        "sample": {
            "dataset": dataset,
            "ground_truth": ground_truth,
        },
        "server_result": {
            "mode": mode,
            "safe": prediction,
            "response": response,
        },
    }


class EvaluateLocalTestResultsTests(unittest.TestCase):
    def test_binary_and_safety_level_metrics(self) -> None:
        records = [
            _record(
                event_id="lab-tp",
                dataset="labsafety_gen",
                ground_truth={"safety_label": "hazardous"},
                prediction="unsafe",
                response="unsafe",
            ),
            _record(
                event_id="lab-fp",
                dataset="labsafety_gen",
                ground_truth={"safety_label": "non-hazardous"},
                prediction="unsafe",
                response="unsafe",
                mode="latency",
            ),
            _record(
                event_id="inspec-tn",
                dataset="inspecsafe_safety_level",
                ground_truth={
                    "overall_safety_level": "Level IV",
                    "hazards": [],
                },
                prediction="safe",
                response='{"overall_safety_level":"Level IV","hazards":[]}',
            ),
            _record(
                event_id="inspec-fn",
                dataset="inspecsafe_safety_level",
                ground_truth={
                    "overall_safety_level": "Level II",
                    "hazards": ["water pooling"],
                },
                prediction="safe",
                response='{"overall_safety_level":"Level IV","hazards":[]}',
            ),
        ]

        report = evaluate_records(records)
        binary = report["binary"]
        self.assertEqual(binary["evaluated"], 4)
        self.assertEqual(binary["correct"], 2)
        self.assertEqual(binary["accuracy"], 0.5)
        self.assertEqual(binary["unsafe_precision"], 0.5)
        self.assertEqual(binary["unsafe_recall"], 0.5)
        self.assertEqual(binary["unsafe_f1"], 0.5)
        self.assertEqual(
            binary["confusion"],
            {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        )
        self.assertEqual(report["by_dataset"]["labsafety_gen"]["accuracy"], 0.5)
        self.assertEqual(report["by_mode"]["accuracy"]["evaluated"], 3)
        self.assertEqual(report["inspecsafe_safety_level"]["accuracy"], 0.5)

    def test_parse_failure_reduces_coverage_and_end_to_end_accuracy(self) -> None:
        records = [
            _record(
                event_id="correct",
                dataset="labsafety_gen",
                ground_truth={"safety_label": "non-hazardous"},
                prediction="safe",
                response="safe",
            ),
            _record(
                event_id="parse-fail",
                dataset="labsafety_gen",
                ground_truth={"safety_label": "hazardous"},
                prediction=None,
                response="unparseable",
            ),
        ]
        binary = evaluate_records(records)["binary"]
        self.assertEqual(binary["evaluated"], 1)
        self.assertEqual(binary["parse_failures"], 1)
        self.assertEqual(binary["coverage"], 0.5)
        self.assertEqual(binary["accuracy"], 1.0)
        self.assertEqual(binary["end_to_end_accuracy"], 0.5)

    def test_jsonl_loader_skips_duplicate_and_invalid_lines(self) -> None:
        record = _record(
            event_id="same-event",
            dataset="labsafety_gen",
            ground_truth={"safety_label": "hazardous"},
            prediction="unsafe",
            response="unsafe",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                json.dumps(record) + "\n"
                + "not-json\n"
                + json.dumps(record) + "\n\n",
                encoding="utf-8",
            )
            report = evaluate_jsonl(path)

        self.assertEqual(report["input"]["records"], 1)
        self.assertEqual(report["input"]["invalid_json_lines"], 1)
        self.assertEqual(report["input"]["duplicate_events"], 1)
        self.assertEqual(report["input"]["blank_lines"], 1)


if __name__ == "__main__":
    unittest.main()
