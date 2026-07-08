"""Reusable two-stage safety inference policy for InspecSafe images.

This module contains no model-loading code. Callers provide a ``generate``
function, which keeps the gate, prompts, and output normalization independently
testable and makes the policy easy to reuse with another VLM backend.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

from config import (
    INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
)


SafetyGenerator = Callable[[Union[str, Path], str, int], str]


def build_stage_one_prompt(query: str) -> str:
    """Build the short classification-only InspecSafe prompt."""
    return f"""{query}

Classify the query image as safe or unsafe.
Respond with exactly one word: safe or unsafe.
"""


def build_stage_two_prompt(query: str) -> str:
    """Build the verification prompt used after an unsafe first pass."""
    return f"""
{query}

Give one short annotation describing the image. And than give a conclusion of safe or unsafe.

Return your answer in exactly this format:
Annotation: <one short sentence>
Final label: safe or unsafe
"""


def parse_stage_one_label(output: str | None) -> str | None:
    """Parse a classification-only response without guessing ambiguous text."""
    if not output:
        return None

    normalized = output.strip().lower().strip("`*_# \t\r\n.,;:!?")
    if normalized in {"safe", "unsafe"}:
        return normalized

    labels = set(re.findall(r"\b(?:safe|unsafe)\b", normalized))
    if len(labels) == 1:
        return labels.pop()
    return None


def parse_stage_two_label(output: str | None) -> str | None:
    """Parse the explicit final label from the verification response."""
    if not output:
        return None

    text = output.strip().lower()
    match = re.search(r"final\s+label\s*:\s*(unsafe|safe)\b", text)
    if match:
        return match.group(1)

    labels = re.findall(r"\b(unsafe|safe)\b", text)
    return labels[-1] if labels else None


def extract_stage_two_annotation(output: str | None) -> str:
    """Extract and normalize the annotation field from a stage-two response."""
    if not output:
        return ""

    match = re.search(
        r"annotation\s*:\s*(.*?)(?=\n\s*final\s+label\s*:|\Z)",
        output.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return " ".join(match.group(1).split())


def run_two_stage_safety_inference(
    query_image: str | Path,
    query: str,
    generate: SafetyGenerator,
    *,
    stage_one_max_new_tokens: int = INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    stage_two_max_new_tokens: int = INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run the two-stage unsafe confirmation policy.

    Only two explicit ``unsafe`` decisions produce an unsafe final result. A
    safe or unparseable response at either gate is normalized to the requested
    final ``safe`` result with an empty annotation.
    """
    if stage_one_max_new_tokens < 1 or stage_two_max_new_tokens < 1:
        raise ValueError("Generation token limits must be positive integers.")

    stage_one_prompt = build_stage_one_prompt(query)
    stage_one_output = generate(
        query_image,
        stage_one_prompt,
        stage_one_max_new_tokens,
    )
    stage_one_label = parse_stage_one_label(stage_one_output)
    stage_one = {
        "prompt": stage_one_prompt,
        "output": stage_one_output,
        "label": stage_one_label,
        "max_new_tokens": stage_one_max_new_tokens,
    }

    if stage_one_label != "unsafe":
        return {
            "label": "safe",
            "annotation": "",
            "output": "safe",
            "stage_one": stage_one,
            "stage_two": None,
        }

    stage_two_prompt = build_stage_two_prompt(query)
    stage_two_output = generate(
        query_image,
        stage_two_prompt,
        stage_two_max_new_tokens,
    )
    stage_two_label = parse_stage_two_label(stage_two_output)
    stage_two = {
        "prompt": stage_two_prompt,
        "output": stage_two_output,
        "label": stage_two_label,
        "max_new_tokens": stage_two_max_new_tokens,
    }

    if stage_two_label != "unsafe":
        return {
            "label": "safe",
            "annotation": "",
            "output": "safe",
            "stage_one": stage_one,
            "stage_two": stage_two,
        }

    return {
        "label": "unsafe",
        "annotation": extract_stage_two_annotation(stage_two_output),
        "output": "unsafe",
        "stage_one": stage_one,
        "stage_two": stage_two,
    }
