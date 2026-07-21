"""Model-free tests for the two-stage InspecSafe decision policy."""

import unittest
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch

import rag_answer
import retrieval_gating
import vlm_inference
from two_stage_inference import run_two_stage_safety_inference


class StubGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[object, str, int]] = []

    def __call__(self, image: object, prompt: str, limit: int) -> str:
        self.calls.append((image, prompt, limit))
        return next(self.outputs)


class TwoStageSafetyInferenceTests(unittest.TestCase):
    def test_balanced_rag_retrieves_once_and_reuses_results(self) -> None:
        top_k_results = [{"image_path": "reference.jpg", "safe_label": "unsafe"}]
        shared_retrieved = [top_k_results[0]]
        stage_messages = [[{"role": "user", "content": "stage"}]]
        search = Mock(return_value=top_k_results)
        retriever_stub = types.ModuleType("retriever")
        retriever_stub.search_by_query_image = search

        with patch.dict(sys.modules, {"retriever": retriever_stub}):
            with patch.object(
                vlm_inference,
                "_resolve_query_image_path",
                return_value=Path("query.jpg"),
            ):
                with patch.object(
                    retrieval_gating,
                    "gate_retrieval_results",
                    return_value=shared_retrieved,
                ) as gate:
                    with patch.object(
                        rag_answer,
                        "build_balanced_two_stage_rag_messages",
                        return_value=stage_messages,
                    ) as build_messages:
                        with patch.object(
                            vlm_inference,
                            "_run_vlm_messages",
                            side_effect=[
                                "unsafe",
                                "Annotation: Missing hard hat.\nFinal label: unsafe",
                            ],
                        ) as generate:
                            result = vlm_inference.VLM_inference_two_stage_with_RAG(
                                "safety judgement",
                                Path("query.jpg"),
                                query="Inspect the image.",
                                top_k=3,
                                gated_rag=0.7,
                                rag_dataset="constructionsite10k",
                                stage_one_max_new_tokens=8,
                                stage_two_max_new_tokens=128,
                            )

        search.assert_called_once_with(
            Path("query.jpg"),
            top_k=3,
            dataset="constructionsite10k",
        )
        gate.assert_called_once_with(top_k_results, 0.7)
        self.assertEqual(build_messages.call_count, 2)
        self.assertIs(build_messages.call_args_list[0].args[2], shared_retrieved)
        self.assertIs(build_messages.call_args_list[1].args[2], shared_retrieved)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(result["retrieved"], shared_retrieved)
        self.assertEqual(result["label"], "unsafe")

    def test_safe_first_stage_skips_second_generation(self) -> None:
        generator = StubGenerator(["safe"])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertEqual(result["annotation"], "")
        self.assertIsNone(result["stage_two"])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0][2], 8)

    def test_second_safe_result_overrides_first_unsafe_result(self) -> None:
        generator = StubGenerator([
            "unsafe",
            "Annotation: A possible issue is visible.\nFinal label: safe",
        ])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertEqual(result["annotation"], "")
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(generator.calls[1][2], 128)

    def test_two_unsafe_results_return_annotation(self) -> None:
        generator = StubGenerator([
            "unsafe",
            "Annotation: Worker is missing a hard hat.\nFinal label: unsafe",
        ])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["output"], "unsafe")
        self.assertEqual(result["label"], "unsafe")
        self.assertEqual(result["annotation"], "Worker is missing a hard hat.")

    def test_unparseable_second_stage_becomes_safe(self) -> None:
        generator = StubGenerator(["unsafe", "The image should be reviewed."])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertEqual(result["annotation"], "")
        self.assertIsNone(result["stage_two"]["label"])

    def test_ambiguous_first_stage_does_not_trigger_second_generation(self) -> None:
        generator = StubGenerator(["safe or unsafe"])

        result = run_two_stage_safety_inference("image.jpg", "Is it safe?", generator)

        self.assertEqual(result["label"], "safe")
        self.assertIsNone(result["stage_one"]["label"])
        self.assertEqual(len(generator.calls), 1)


if __name__ == "__main__":
    unittest.main()
