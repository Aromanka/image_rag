"""Tests for post-top-k similarity gating."""

import unittest
import sys
import types
from unittest.mock import patch

from retrieval_gating import gate_retrieval_results


class RetrievalGatingTests(unittest.TestCase):
    def test_filters_after_receiving_ranked_top_k_results(self) -> None:
        results = [
            {"id": "a", "distance": 0.1},
            {"id": "b", "distance": 0.4},
            {"id": "c", "distance": 1.2},
        ]

        gated = gate_retrieval_results(results, 0.6)

        self.assertEqual([item["id"] for item in gated], ["a", "b"])
        self.assertAlmostEqual(gated[0]["similarity"], 0.9)
        self.assertAlmostEqual(gated[1]["similarity"], 0.6)

    def test_default_zero_allows_no_results(self) -> None:
        results = [{"id": "negative", "distance": 1.1}]

        self.assertEqual(gate_retrieval_results(results, 0.0), [])

    def test_preserves_existing_similarity_and_order(self) -> None:
        results = [
            {"id": "first", "similarity": 0.8},
            {"id": "second", "similarity": 0.9},
        ]

        gated = gate_retrieval_results(results, 0.5)

        self.assertEqual([item["id"] for item in gated], ["first", "second"])

    def test_rag_pipeline_accepts_zero_results_after_top_k(self) -> None:
        import vlm_inference

        fake_retriever = types.ModuleType("retriever")
        fake_retriever.search_by_query_image = lambda *args, **kwargs: [
            {"id": "a", "distance": 0.4, "similarity": 0.6},
            {"id": "b", "distance": 0.5, "similarity": 0.5},
        ]
        image_path = "demo/inspecsafe_rag_details/61/query_image.jpg"

        with patch.dict(sys.modules, {"retriever": fake_retriever}):
            with patch.object(
                vlm_inference,
                "_run_vlm_messages",
                return_value="safe",
            ):
                result = vlm_inference.VLM_inference_with_RAG(
                    "safety judgement",
                    image_path,
                    top_k=2,
                    gated_rag=0.8,
                )

        self.assertEqual(result["retrieved_count_before_gate"], 2)
        self.assertEqual(result["retrieved_count"], 0)
        self.assertEqual(result["retrieved"], [])
        self.assertEqual(result["output"], "safe")


if __name__ == "__main__":
    unittest.main()
