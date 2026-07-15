"""Model-free tests for image-server mode routing."""

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import image_server
from config import INSPECSAFE_DATASET, SAFETY_JUDGEMENT_TASK, SAFETY_LEVEL_TASK


class ImageServerModeTests(unittest.TestCase):
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
