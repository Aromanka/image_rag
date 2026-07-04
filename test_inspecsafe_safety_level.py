"""CPU-only tests for InspecSafe safety-level parsing and metrics."""

import unittest
from pathlib import Path

from utils.evaluate_utils import (
    evaluate_inspecsafe_safety_level_results_json,
    extract_inspecsafe_safety_level_json,
    normalize_inspecsafe_safety_level,
)
from rag_answer import build_inspecsafe_safety_level_rag_messages
from vlm_inference import build_baseline_prompt
from evaluate_inspecsafe_safety_level import pipeline_image_to_dataset_path


class SafetyLevelParsingTests(unittest.TestCase):
    def test_converts_pipeline_anomaly_path_to_original_dataset(self):
        converted = pipeline_image_to_dataset_path(
            "images/test__oil_chemical-Level01-Wheeled-002319__"
            "oil_chemical-Level01-Wheeled-002319-001.jpg",
            "/root/autodl-tmp/data/inspecsafe/DATA_PATH",
        )
        self.assertEqual(
            converted,
            Path("/root/autodl-tmp/data/inspecsafe/DATA_PATH")
            / "test"
            / "Annotations"
            / "Anomaly_data"
            / "oil_chemical-Level01-Wheeled-002319"
            / "oil_chemical-Level01-Wheeled-002319-001.jpg",
        )

    def test_converts_level04_to_normal_data(self):
        converted = pipeline_image_to_dataset_path(
            "images/train__coal_conveyor-Level04-SuspendedRail-000001__"
            "coal_conveyor-Level04-SuspendedRail-000001-001.jpg",
            "/dataset",
        )
        self.assertEqual(
            converted,
            Path("/dataset/train/Annotations/Normal_data/")
            / "coal_conveyor-Level04-SuspendedRail-000001"
            / "coal_conveyor-Level04-SuspendedRail-000001-001.jpg",
        )

    def test_rejects_unrecognized_pipeline_path(self):
        with self.assertRaises(ValueError):
            pipeline_image_to_dataset_path("images/not-a-pipeline-name.jpg", "/dataset")

    def test_extracts_and_repairs_first_json_object(self):
        parsed = extract_inspecsafe_safety_level_json(
            "prefix {'hazards': ['smoke',], "
            "'overall_safety_level': 'Level I',} suffix"
        )
        self.assertEqual(parsed["hazards"], ["smoke"])
        self.assertEqual(parsed["overall_safety_level"], "Level I")

    def test_normalizes_supported_level_forms(self):
        self.assertEqual(normalize_inspecsafe_safety_level("Level IV"), "Level IV")
        self.assertEqual(normalize_inspecsafe_safety_level("3"), "Level III")
        self.assertEqual(normalize_inspecsafe_safety_level("level two"), "Level II")
        self.assertIsNone(normalize_inspecsafe_safety_level("unknown"))

    def test_rag_recovers_level_from_existing_binary_index_path(self):
        messages = build_inspecsafe_safety_level_rag_messages(
            "assess",
            "query.jpg",
            [{
                "image_path": "images/oil_gas-Level02-Wheeled-1/image.jpg",
                "caption": "Water is pooling on the floor.",
                "safe_label": "unsafe",
            }],
        )
        reference_text = messages[1]["content"][1]["text"]
        self.assertIn("Reference label: Level II", reference_text)

    def test_safety_level_task_alias(self):
        prompt = build_baseline_prompt("safety_level")
        self.assertIn("overall_safety_level", prompt)


class SafetyLevelMetricTests(unittest.TestCase):
    def test_reference_metric_semantics(self):
        payload = {
            "results": [
                {
                    "ground_truth": {
                        "scene_description": "Smoke near equipment.",
                        "hazards": ["smoke", "no mask"],
                        "overall_safety_level": "Level I",
                    },
                    "output": (
                        '{"scene_description":"Smoke.","hazards":["smoke"],'
                        '"overall_safety_level":"Level I"}'
                    ),
                },
                {
                    "ground_truth": {
                        "scene_description": "Water on floor.",
                        "hazards": ["water pooling"],
                        "overall_safety_level": "Level II",
                    },
                    "output": (
                        '{"scene_description":"Debris.",'
                        '"hazards":["foreign objects"],'
                        '"overall_safety_level":"Level III"}'
                    ),
                },
                {
                    "ground_truth": {
                        "scene_description": "Normal scene.",
                        "hazards": [],
                        "overall_safety_level": "Level IV",
                    },
                    "output": "not JSON",
                },
            ]
        }

        evaluated = evaluate_inspecsafe_safety_level_results_json(
            payload,
            compute_scene_metrics=False,
        )
        summary = evaluated["summary"]

        self.assertEqual(summary["n_samples"], 3)
        self.assertAlmostEqual(summary["json_parse_rate"], 2 / 3)
        self.assertAlmostEqual(summary["level_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["level_macro_f1"], 1 / 3)
        self.assertAlmostEqual(summary["level_micro_p"], 1 / 2)
        self.assertAlmostEqual(summary["level_micro_r"], 1 / 3)
        self.assertAlmostEqual(summary["level_micro_f1"], 0.4)
        self.assertAlmostEqual(summary["hazard_precision"], 1 / 2)
        self.assertAlmostEqual(summary["hazard_recall"], 1 / 3)
        self.assertAlmostEqual(summary["hazard_f1"], 0.4)
        self.assertIsNone(summary["scene_sbert_sim"])
        self.assertEqual(evaluated["results"][2]["status"], "PARSE_FAIL")


if __name__ == "__main__":
    unittest.main()
