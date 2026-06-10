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


def parse_constructionsite10k_output(text: str | None) -> tuple[dict[str, Any], bool]:
    """Parse a ConstructionSite-10K JSON response from model text."""
    if not text:
        return {"annotation": "", "violations": []}, False

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "violations" in parsed:
            return parsed, True
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "violations" in parsed:
            return parsed, True

    return {"annotation": "", "violations": []}, False


def get_violation_rules(violations: Any) -> set[int]:
    rules: set[int] = set()
    if not isinstance(violations, list):
        return rules

    for violation in violations:
        if not isinstance(violation, dict):
            continue
        try:
            rule = int(violation.get("rule"))
        except (TypeError, ValueError):
            continue
        if 1 <= rule <= 4:
            rules.add(rule)
    return rules


def evaluate_constructionsite10k_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved ConstructionSite-10K inference results."""
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

    all_rules = [1, 2, 3, 4]
    tp = {rule: 0 for rule in all_rules}
    fp = {rule: 0 for rule in all_rules}
    fn = {rule: 0 for rule in all_rules}
    parse_failures = 0
    valid_results: list[dict[str, Any]] = []

    for sample in results:
        if not isinstance(sample, dict):
            parse_failures += 1
            continue

        gt_text = sample.get("ground_truth_output") or sample.get("ground_truth") or ""
        pred_text = sample.get("output") or sample.get("pred_raw") or ""

        gt_json, _ = parse_constructionsite10k_output(gt_text)
        gt_rules = get_violation_rules(gt_json.get("violations", []))

        sample["gt_annotation"] = gt_json.get("annotation", "")
        sample["gt_rules"] = sorted(gt_rules)

        if sample.get("error"):
            sample["pred_annotation"] = ""
            sample["pred_rules"] = []
            sample["parse_failed"] = True
            sample["status"] = "ERROR"
            parse_failures += 1
            continue

        pred_json, parse_ok = parse_constructionsite10k_output(pred_text)
        pred_rules = get_violation_rules(pred_json.get("violations", []))

        sample["pred_annotation"] = pred_json.get("annotation", "") if parse_ok else ""
        sample["pred_rules"] = sorted(pred_rules) if parse_ok else []
        sample["parse_failed"] = not parse_ok

        if not parse_ok:
            sample["status"] = "PARSE_FAIL"
            parse_failures += 1
            continue

        for rule in all_rules:
            pred_pos = rule in pred_rules
            gt_pos = rule in gt_rules
            if pred_pos and gt_pos:
                tp[rule] += 1
            elif pred_pos and not gt_pos:
                fp[rule] += 1
            elif not pred_pos and gt_pos:
                fn[rule] += 1

        sample["status"] = "CORRECT" if pred_rules == gt_rules else "WRONG"
        valid_results.append(sample)

    per_rule = {}
    for rule in all_rules:
        precision = tp[rule] / (tp[rule] + fp[rule]) if tp[rule] + fp[rule] else 0.0
        recall = tp[rule] / (tp[rule] + fn[rule]) if tp[rule] + fn[rule] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_rule[str(rule)] = {
            "tp": tp[rule],
            "fp": fp[rule],
            "fn": fn[rule],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    valid_count = len(valid_results)
    exact_match = sum(
        1 for sample in valid_results if sample["gt_rules"] == sample["pred_rules"]
    )
    safe_correct = sum(
        1
        for sample in valid_results
        if (len(sample["gt_rules"]) == 0) == (len(sample["pred_rules"]) == 0)
    )
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    data["summary"] = {
        "total_samples": len(results),
        "valid_samples": valid_count,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(results) if results else 0.0,
        "exact_match_acc": exact_match / valid_count if valid_count else 0.0,
        "safe_unsafe_acc": safe_correct / valid_count if valid_count else 0.0,
        "macro_precision": sum(per_rule[str(r)]["precision"] for r in all_rules)
        / len(all_rules),
        "macro_recall": sum(per_rule[str(r)]["recall"] for r in all_rules)
        / len(all_rules),
        "macro_f1": sum(per_rule[str(r)]["f1"] for r in all_rules) / len(all_rules),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }
    data["per_rule"] = per_rule

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
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "inspecsafe", "constructionsite10k"],
        default="auto",
        help="Evaluation metric type for the saved JSON.",
    )
    return parser.parse_args()


def _detect_results_type(results_json: Path) -> str:
    with results_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    results = payload.get("results", []) if isinstance(payload, dict) else payload
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict) and "ground_truth_output" in first:
            return "constructionsite10k"
    return "inspecsafe"


if __name__ == "__main__":
    args = _parse_args()
    dataset_type = (
        _detect_results_type(args.results_json)
        if args.dataset_type == "auto"
        else args.dataset_type
    )
    if dataset_type == "constructionsite10k":
        evaluated = evaluate_constructionsite10k_results_json(
            args.results_json,
            args.output_json,
        )
        summary = evaluated["summary"]
        print(f"Total samples:  {summary['total_samples']}")
        print(f"Valid samples:  {summary['valid_samples']}")
        print(f"Parse failures: {summary['parse_failures']}")
        print(f"Exact match:    {summary['exact_match_acc']:.4f}")
        print(f"Safe/unsafe:    {summary['safe_unsafe_acc']:.4f}")
        print(f"Macro F1:       {summary['macro_f1']:.4f}")
        print(f"Micro F1:       {summary['micro_f1']:.4f}")
    else:
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
