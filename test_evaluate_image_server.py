"""Model-free tests for the direct image-server evaluator."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import evaluate_image_server as evaluator


def _record(truth: str, predicted: str, *, score: float = 0.5) -> dict:
    return {
        "status": "success",
        "ground_truth": truth,
        "backend_seconds": 1.0,
        "sbert_score": score,
        "server_result": {"safe": predicted},
    }


class EvaluateImageServerTests(unittest.TestCase):
    def test_constructionsite_default_image_prefix(self) -> None:
        image_path = evaluator.constructionsite_image_path(
            {"image": "images\\0000001.jpg"},
            Path("constructionsite_10k/test.json"),
            None,
        )

        self.assertEqual(
            image_path,
            evaluator.PROJECT_ROOT
            / "data"
            / "constructionsite"
            / "images"
            / "0000001.jpg",
        )

    def test_binary_metrics_use_unsafe_as_positive_label(self) -> None:
        records = [
            _record("unsafe", "unsafe", score=0.8),
            _record("safe", "unsafe", score=0.6),
            _record("safe", "safe", score=0.4),
            _record("unsafe", "safe", score=0.2),
            {
                "status": "error",
                "ground_truth": "unsafe",
                "backend_seconds": 0.1,
                "sbert_score": None,
            },
        ]

        summary = evaluator.calculate_binary_metrics(records)

        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["fp"], 1)
        self.assertEqual(summary["tn"], 1)
        self.assertEqual(summary["fn"], 1)
        self.assertEqual(summary["precision"], 0.5)
        self.assertEqual(summary["recall"], 0.5)
        self.assertEqual(summary["f1"], 0.5)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["end_to_end_accuracy"], 0.4)
        self.assertAlmostEqual(summary["sbert_score"], 0.5)
        self.assertEqual(summary["errors"], 1)

    def test_direct_inference_uses_image_server_dispatch_and_parser(self) -> None:
        sample = evaluator.EvaluationSample(
            sample_id="sample-1",
            image_path=Path("query.jpg"),
            ground_truth="safe",
            reference_text="No hazards are visible.",
        )
        raw_result = {
            "output": '{"scene_description":"Normal scene.","hazards":[]}',
            "retrieved_count_before_gate": 5,
            "retrieved_count": 2,
        }

        with patch.object(
            evaluator.image_server, "_run_inference", return_value=(raw_result["output"], raw_result)
        ) as inference:
            records = evaluator._evaluate_mode(
                samples=[sample],
                mode=evaluator.image_server.ACCURACY_MODE,
                top_k=7,
                max_new_tokens=384,
                stage_one_max_new_tokens=8,
                stage_two_max_new_tokens=128,
                checkpoint_every=0,
                checkpoint=lambda _: None,
            )

        inference.assert_called_once_with(
            image_path=Path("query.jpg"),
            mode=evaluator.image_server.ACCURACY_MODE,
            top_k=7,
            max_new_tokens=384,
            stage_one_max_new_tokens=8,
            stage_two_max_new_tokens=128,
        )
        self.assertEqual(records[0]["server_result"]["safe"], "safe")
        self.assertEqual(records[0]["server_result"]["annotation"], "Normal scene.")
        self.assertTrue(records[0]["correct"])

    def test_inspecsafe_loader_maps_labels_and_reference_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text(
                "id,image_path,caption,safe_label\n"
                "1,relative/a.jpg,Normal operation.,safe\n"
                "2,relative/b.jpg,Smoke is visible.,unsafe\n",
                encoding="utf-8",
            )
            samples = evaluator.load_inspecsafe_samples(
                csv_path, limit=1, offset=1, image_root=Path(temp_dir)
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].sample_id, "2")
        self.assertEqual(samples[0].ground_truth, "unsafe")
        self.assertEqual(samples[0].reference_text, "Smoke is visible.")

    def test_labsafety_reference_prefers_hazards_for_unsafe_scene(self) -> None:
        self.assertEqual(
            evaluator._labsafety_reference_text(
                {"hazards": ["missing goggles"], "description": "Scene text"}
            ),
            "missing goggles",
        )
        self.assertEqual(
            evaluator._labsafety_reference_text(
                {"hazards": [], "description": "No hazards are visible."}
            ),
            "No hazards are visible.",
        )


if __name__ == "__main__":
    unittest.main()
