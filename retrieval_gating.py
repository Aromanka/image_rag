"""Similarity-threshold gating for already-ranked retrieval results."""

from __future__ import annotations

import math
from typing import Any


def validate_gated_rag(gated_rag: float) -> float:
    """Return a finite floating-point similarity threshold."""
    threshold = float(gated_rag)
    if not math.isfinite(threshold):
        raise ValueError("gated_rag must be a finite number.")
    return threshold


def gate_retrieval_results(
    results: list[dict[str, Any]],
    gated_rag: float,
) -> list[dict[str, Any]]:
    """Filter top-k results whose cosine similarity is below the threshold.

    Chroma cosine collections return distance, so similarity is ``1-distance``.
    The input order is preserved and the returned dictionaries expose the
    computed ``similarity`` for debugging and evaluation artifacts.
    """
    threshold = validate_gated_rag(gated_rag)
    gated: list[dict[str, Any]] = []

    for result in results:
        similarity = result.get("similarity")
        if similarity is None:
            if result.get("distance") is None:
                raise ValueError(
                    "Retrieval result must contain 'similarity' or 'distance'."
                )
            similarity = 1.0 - float(result["distance"])
        else:
            similarity = float(similarity)

        if not math.isfinite(similarity):
            raise ValueError("Retrieval similarity must be a finite number.")
        if similarity >= threshold:
            gated.append({**result, "similarity": similarity})

    return gated
