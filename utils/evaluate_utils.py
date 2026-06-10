"""Utilities for evaluating saved VLM result JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def extract_label(output: str | None) -> str | None:
    """Extract safe/unsafe label from VLM output text."""
    if not output:
        return None

    text = output.strip().lower()
    match = re.search(r"final\s+label\s*:\s*(safe|unsafe)", text)
    if match:
        return match.group(1)

    matches = re.findall(r"\b(unsafe|safe)\b", text)
    if matches:
        return matches[-1]

    return None


def evaluate_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved inference results and optionally write updated JSON.

    The input can be a path to a JSON file, a top-level JSON object containing a
    ``results`` list, or a raw list of sample result dictionaries. Each sample is
    updated with ``predicted`` and ``status`` fields.
    """
    source_path: Path | None = None
    if isinstance(results_json, (str, Path)):
        source_path = Path(results_json)
        with source_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] | list[dict[str, Any]] = json.load(file)
    else:
        payload = results_json

    if isinstance(payload, list):
        data: dict[str, Any] = {"results": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("results_json must be a path, dict, or list of dictionaries.")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Results JSON must contain a 'results' list.")

    total = len(results)
    correct = 0
    errors = 0

    for sample in results:
        if not isinstance(sample, dict):
            errors += 1
            continue

        ground_truth = str(sample.get("ground_truth", "")).strip().lower()
        sample["ground_truth"] = ground_truth

        if sample.get("error"):
            sample["predicted"] = None
            sample["status"] = "ERROR"
            errors += 1
            continue

        if ground_truth not in ("safe", "unsafe"):
            sample["predicted"] = None
            sample["status"] = "SKIP"
            errors += 1
            continue

        predicted = extract_label(sample.get("output"))
        sample["predicted"] = predicted

        if predicted is None:
            sample["status"] = "PARSE_FAIL"
            errors += 1
        elif predicted == ground_truth:
            sample["status"] = "CORRECT"
            correct += 1
        else:
            sample["status"] = "WRONG"

    evaluated = total - errors
    data["summary"] = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "errors_or_skipped": errors,
        "accuracy": correct / evaluated if evaluated > 0 else 0.0,
    }

    if output_json is not None:
        target_path = Path(output_json)
    else:
        target_path = source_path

    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved result JSON.")
    parser.add_argument("results_json", type=Path, help="Path to saved results JSON.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the evaluated JSON. Defaults to overwriting input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluated = evaluate_results_json(args.results_json, args.output_json)
    summary = evaluated["summary"]
    print(f"Total samples:  {summary['total']}")
    print(f"Evaluated:      {summary['evaluated']}")
    print(f"Correct:        {summary['correct']}")
    print(f"Errors/Skipped: {summary['errors_or_skipped']}")
    print(
        "Accuracy:       "
        f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
    )
