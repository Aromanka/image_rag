"""Model-free tests for image-server mode routing."""

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import image_server
from config import (
    INSPECSAFE_DATASET,
    SAFETY_JUDGEMENT_TASK,
    SAFETY_LEVEL_TASK,
    VLM_LORA_WEIGHTS,
)


class ImageServerModeTests(unittest.TestCase):
    def test_server_defaults_to_configured_lora_for_latency_first(self) -> None:
        with patch.dict(image_server.os.environ, {"VLM_LORA_WEIGHTS": ""}):
            with patch.object(image_server, "configure_lora_weights") as configure:
                selected = image_server._configure_server_lora(None)

        self.assertEqual(selected, VLM_LORA_WEIGHTS)
        configure.assert_called_once_with(VLM_LORA_WEIGHTS)

    def test_server_preserves_environment_lora_default(self) -> None:
        with patch.dict(
            image_server.os.environ,
            {"VLM_LORA_WEIGHTS": "environment-lora"},
        ):
            with patch.object(image_server, "configure_lora_weights") as configure:
                selected = image_server._configure_server_lora(None)

        self.assertEqual(selected, "environment-lora")
        configure.assert_called_once_with("environment-lora")

    def test_explicit_lora_override_is_preserved(self) -> None:
        override = Path("custom-lora")
        with patch.object(image_server, "configure_lora_weights") as configure:
            selected = image_server._configure_server_lora(override)

        self.assertEqual(selected, override)
        configure.assert_called_once_with(override)

    def test_normalizes_four_modes_and_first_aliases(self) -> None:
        expected = {
            "accuracy": image_server.ACCURACY_MODE,
            "accuracy-first": image_server.ACCURACY_MODE,
            "latency": image_server.LATENCY_MODE,
            "latency-first": image_server.LATENCY_MODE,
            "energy": image_server.ENERGY_MODE,
            "energy-first": image_server.ENERGY_MODE,
            "balanced": image_server.BALANCED_MODE,
            "balanced-mode": image_server.BALANCED_MODE,
        }
        for supplied, normalized in expected.items():
            with self.subTest(mode=supplied):
                self.assertEqual(image_server._normalize_mode(supplied), normalized)

        with self.assertRaises(ValueError):
            image_server._normalize_mode("unknown")

    def test_accuracy_energy_and_balanced_use_inspecsafe_level_rag(self) -> None:
        rag_result = {"output": '{"overall_safety_level":"Level IV"}'}
        for mode in image_server.RAG_MODES:
            with self.subTest(mode=mode):
                with patch.object(
                    image_server,
                    "VLM_inference_with_RAG",
                    return_value=rag_result,
                ) as rag_inference:
                    output, result = image_server._run_inference(
                        image_path=Path("query.jpg"),
                        mode=mode,
                        top_k=7,
                        max_new_tokens=384,
                        stage_one_max_new_tokens=8,
                        stage_two_max_new_tokens=128,
                    )

                self.assertEqual(output, rag_result["output"])
                self.assertIs(result, rag_result)
                rag_inference.assert_called_once_with(
                    SAFETY_LEVEL_TASK,
                    Path("query.jpg"),
                    query=image_server.ACCURACY_QUERY,
                    top_k=7,
                    gated_rag=image_server.ACCURACY_GATE,
                    rag_dataset=INSPECSAFE_DATASET,
                    max_new_tokens=384,
                )

    def test_latency_mode_keeps_existing_two_stage_path(self) -> None:
        latency_result = {"label": "unsafe", "annotation": "Smoke is visible."}
        with patch.object(
            image_server,
            "VLM_inference_two_stage",
            return_value=latency_result,
        ) as two_stage:
            with patch.object(image_server, "VLM_inference_with_RAG") as rag_inference:
                output, result = image_server._run_inference(
                    image_path=Path("query.jpg"),
                    mode=image_server.LATENCY_MODE,
                    top_k=7,
                    max_new_tokens=384,
                    stage_one_max_new_tokens=8,
                    stage_two_max_new_tokens=128,
                )

        self.assertEqual(output, "unsafe Smoke is visible.")
        self.assertIs(result, latency_result)
        rag_inference.assert_not_called()
        two_stage.assert_called_once_with(
            SAFETY_JUDGEMENT_TASK,
            Path("query.jpg"),
            query=image_server.SAFETY_PROMPT,
            stage_one_max_new_tokens=8,
            stage_two_max_new_tokens=128,
        )

    def test_accuracy_parser_uses_hazards_and_scene_description(self) -> None:
        safe_output = (
            '{"scene_description":"No hazards are visible.",'
            '"hazards":[],"overall_safety_level":"Level IV"}'
        )
        unsafe_output = (
            'prefix {"scene_description":"Smoke is visible.",'
            '"hazards":["smoke"],"overall_safety_level":"Level I"} suffix'
        )

        self.assertEqual(
            image_server._parse_accuracy_response(safe_output),
            {"safe": "safe", "annotation": "No hazards are visible."},
        )
        self.assertEqual(
            image_server._parse_accuracy_response(unsafe_output),
            {"safe": "unsafe", "annotation": "Smoke is visible."},
        )

    def test_accuracy_parser_treats_missing_or_invalid_hazards_as_unsafe(self) -> None:
        self.assertEqual(
            image_server._parse_accuracy_response(
                '{"scene_description":"Unstructured hazards."}'
            ),
            {"safe": "unsafe", "annotation": "Unstructured hazards."},
        )
        self.assertEqual(
            image_server._parse_accuracy_response("not JSON"),
            {"safe": "unsafe", "annotation": ""},
        )

    def test_latency_parser_removes_only_one_leading_label(self) -> None:
        cases = {
            "safe": {"safe": "safe", "annotation": ""},
            "SAFE   Normal operation.": {
                "safe": "safe",
                "annotation": "Normal operation.",
            },
            "unsafe unsafe Smoke is visible.": {
                "safe": "unsafe",
                "annotation": "unsafe Smoke is visible.",
            },
            "Smoke is visible.": {
                "safe": "unsafe",
                "annotation": "Smoke is visible.",
            },
        }
        for output, expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(image_server._parse_latency_response(output), expected)

    def test_each_mode_has_an_independent_parser_entry(self) -> None:
        expected_parsers = {
            image_server.ACCURACY_MODE: image_server._parse_accuracy_response,
            image_server.LATENCY_MODE: image_server._parse_latency_response,
            image_server.ENERGY_MODE: image_server._parse_energy_response,
            image_server.BALANCED_MODE: image_server._parse_balanced_response,
        }
        self.assertEqual(image_server.MODE_RESPONSE_PARSERS, expected_parsers)
        self.assertIsNot(
            image_server.MODE_RESPONSE_PARSERS[image_server.ENERGY_MODE],
            image_server.MODE_RESPONSE_PARSERS[image_server.ACCURACY_MODE],
        )
        self.assertIsNot(
            image_server.MODE_RESPONSE_PARSERS[image_server.BALANCED_MODE],
            image_server.MODE_RESPONSE_PARSERS[image_server.ACCURACY_MODE],
        )

    def test_success_payload_contains_unified_semantic_fields(self) -> None:
        with patch.object(
            image_server,
            "active_lora_weights",
            return_value="configured-lora",
        ):
            payload = image_server._build_success_response_payload(
                image_server.ACCURACY_MODE,
                '{"scene_description":"Normal scene.","hazards":[]}',
                {"retrieved_count_before_gate": 5, "retrieved_count": 2},
                1.2345,
            )

        self.assertEqual(payload["safe"], "safe")
        self.assertEqual(payload["annotation"], "Normal scene.")
        self.assertEqual(payload["lora_weights"], "configured-lora")
        self.assertEqual(payload["response_time_seconds"], 1.234)
        self.assertEqual(payload["retrieved_count_before_gate"], 5)
        self.assertEqual(payload["retrieved_count"], 2)

    def test_accuracy_system_prompt_matches_finetuned_evaluator(self) -> None:
        project_root = Path(__file__).resolve().parent

        def literal_assignment(path: Path, name: str) -> str:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                    return ast.literal_eval(node.value)
            raise AssertionError(f"Assignment {name!r} not found in {path}")

        evaluator_prompt = literal_assignment(
            project_root / "evaluate_finetuned_inspecsafe.py",
            "SYSTEM_PROMPT",
        )
        rag_prompt = literal_assignment(
            project_root / "rag_answer.py",
            "INSPECSAFE_SAFETY_LEVEL_SYSTEM_PROMPT",
        )
        self.assertEqual(rag_prompt, evaluator_prompt)


if __name__ == "__main__":
    unittest.main()
