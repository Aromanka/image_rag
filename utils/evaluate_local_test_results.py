"""Evaluate JSONL records produced by ``local_test_display.py``."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluate_utils import (  # noqa: E402
    extract_inspecsafe_safety_level_json,
    normalize_inspecsafe_safety_level,
)
from utils.local_test_data import (  # noqa: E402
    INSPECSAFE_SAFETY_LEVEL_DATASET,
    LABSAFETY_GEN_DATASET,
    normalize_dataset,
)


SAFE = "safe"
UNSAFE = "unsafe"


def load_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load valid JSON objects and report malformed/duplicate input lines."""
    if not path.is_file():
        raise FileNotFoundError(f"Local-test JSONL not found: {path}")

    records: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    stats = {"blank_lines": 0, "invalid_json_lines": 0, "duplicate_events": 0}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                stats["blank_lines"] += 1
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                stats["invalid_json_lines"] += 1
                continue
            if not isinstance(record, dict):
                stats["invalid_json_lines"] += 1
                continue
            event_id = str(record.get("event_id", "")).strip()
            if event_id and event_id in seen_event_ids:
                stats["duplicate_events"] += 1
                continue
            if event_id:
                seen_event_ids.add(event_id)
            records.append(record)
    return records, stats


def evaluate_records(
    records: Iterable[dict[str, Any]],
    *,
    source: str | None = None,
    input_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute binary and InspecSafe level metrics from display records."""
    materialized = list(records)
    overall_rows: list[tuple[str | None, str | None]] = []
    dataset_rows: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    mode_rows: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    association_counts: Counter[str] = Counter()
    level_rows: list[tuple[str | None, str | None]] = []

    for record in materialized:
        dataset = _record_dataset(record)
        truth = _ground_truth_label(record, dataset)
        prediction = _prediction_label(record)
        row = (truth, prediction)
        overall_rows.append(row)
        dataset_rows[dataset or "unknown"].append(row)

        server_result = record.get("server_result")
        mode = (
            str(server_result.get("mode", "unknown"))
            if isinstance(server_result, dict)
            else "unknown"
        )
        mode_rows[mode or "unknown"].append(row)
        association_counts[str(record.get("association", "unknown"))] += 1

        if dataset == INSPECSAFE_SAFETY_LEVEL_DATASET:
            level_rows.append(
                (
                    _ground_truth_level(record),
                    _prediction_level(record),
                )
            )

    return {
        "source": source,
        "input": {
            "records": len(materialized),
            **(input_stats or {}),
        },
        "binary": _binary_metrics(overall_rows),
        "by_dataset": {
            key: _binary_metrics(rows) for key, rows in sorted(dataset_rows.items())
        },
        "by_mode": {
            key: _binary_metrics(rows) for key, rows in sorted(mode_rows.items())
        },
        "inspecsafe_safety_level": _classification_metrics(level_rows),
        "association": dict(sorted(association_counts.items())),
    }


def evaluate_jsonl(path: Path) -> dict[str, Any]:
    records, input_stats = load_jsonl_records(path)
    return evaluate_records(records, source=str(path.resolve()), input_stats=input_stats)


def _record_dataset(record: dict[str, Any]) -> str | None:
    sample = record.get("sample")
    raw_dataset = sample.get("dataset") if isinstance(sample, dict) else None
    raw_dataset = raw_dataset or record.get("dataset")
    if not raw_dataset:
        return None
    try:
        return normalize_dataset(str(raw_dataset))
    except ValueError:
        return str(raw_dataset).strip().lower() or None


def _ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample")
    value = sample.get("ground_truth") if isinstance(sample, dict) else None
    if value is None:
        value = record.get("ground_truth")
    return value if isinstance(value, dict) else {}


def _ground_truth_label(record: dict[str, Any], dataset: str | None) -> str | None:
    ground_truth = _ground_truth(record)
    if dataset == LABSAFETY_GEN_DATASET:
        return _normalize_binary(ground_truth.get("safety_label"))
    if dataset == INSPECSAFE_SAFETY_LEVEL_DATASET:
        level = normalize_inspecsafe_safety_level(
            ground_truth.get("overall_safety_level")
        )
        if level is not None:
            return SAFE if level == "Level IV" else UNSAFE
        hazards = ground_truth.get("hazards")
        if isinstance(hazards, list):
            return SAFE if not hazards else UNSAFE
    for key in ("safe", "safety_label", "label"):
        normalized = _normalize_binary(ground_truth.get(key))
        if normalized is not None:
            return normalized
    return None


def _prediction_label(record: dict[str, Any]) -> str | None:
    result = record.get("server_result")
    if not isinstance(result, dict):
        return None
    return _normalize_binary(result.get("safe"))


def _normalize_binary(value: Any) -> str | None:
    if isinstance(value, bool):
        return SAFE if value else UNSAFE
    normalized = str(value or "").strip().lower().replace("_", "-")
    normalized = " ".join(normalized.split())
    if normalized in {"safe", "non-hazardous", "non hazardous", "normal"}:
        return SAFE
    if normalized in {"unsafe", "hazardous", "hazard", "anomaly", "anomalous"}:
        return UNSAFE
    return None


def _ground_truth_level(record: dict[str, Any]) -> str | None:
    return normalize_inspecsafe_safety_level(
        _ground_truth(record).get("overall_safety_level")
    )


def _prediction_level(record: dict[str, Any]) -> str | None:
    result = record.get("server_result")
    if not isinstance(result, dict):
        return None
    parsed = extract_inspecsafe_safety_level_json(result.get("response"))
    if parsed is None:
        return None
    return normalize_inspecsafe_safety_level(parsed.get("overall_safety_level"))


def _binary_metrics(rows: Iterable[tuple[str | None, str | None]]) -> dict[str, Any]:
    materialized = list(rows)
    tp = fp = tn = fn = 0
    missing_truth = parse_failures = 0
    for truth, prediction in materialized:
        if truth not in {SAFE, UNSAFE}:
            missing_truth += 1
            continue
        if prediction not in {SAFE, UNSAFE}:
            parse_failures += 1
            continue
        if truth == UNSAFE and prediction == UNSAFE:
            tp += 1
        elif truth == SAFE and prediction == UNSAFE:
            fp += 1
        elif truth == SAFE and prediction == SAFE:
            tn += 1
        else:
            fn += 1

    evaluated = tp + fp + tn + fn
    correct = tp + tn
    eligible = evaluated + parse_failures
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_denominator = precision + recall
    return {
        "total_records": len(materialized),
        "eligible": eligible,
        "evaluated": evaluated,
        "correct": correct,
        "missing_truth": missing_truth,
        "parse_failures": parse_failures,
        "coverage": _divide(evaluated, eligible),
        "accuracy": _divide(correct, evaluated),
        "end_to_end_accuracy": _divide(correct, eligible),
        "unsafe_precision": precision,
        "unsafe_recall": recall,
        "unsafe_f1": (
            2 * precision * recall / f1_denominator if f1_denominator else 0.0
        ),
        "safe_specificity": _divide(tn, tn + fp),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _classification_metrics(
    rows: Iterable[tuple[str | None, str | None]],
) -> dict[str, Any]:
    materialized = list(rows)
    missing_truth = parse_failures = correct = evaluated = 0
    for truth, prediction in materialized:
        if truth is None:
            missing_truth += 1
        elif prediction is None:
            parse_failures += 1
        else:
            evaluated += 1
            correct += int(truth == prediction)
    eligible = evaluated + parse_failures
    return {
        "total_records": len(materialized),
        "eligible": eligible,
        "evaluated": evaluated,
        "correct": correct,
        "missing_truth": missing_truth,
        "parse_failures": parse_failures,
        "coverage": _divide(evaluated, eligible),
        "accuracy": _divide(correct, evaluated),
        "end_to_end_accuracy": _divide(correct, eligible),
    }


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def print_report(report: dict[str, Any]) -> None:
    binary = report["binary"]
    confusion = binary["confusion"]
    print("Local Test Accuracy")
    print("=" * 60)
    if report.get("source"):
        print(f"Source:              {report['source']}")
    print(f"Records:             {report['input']['records']}")
    print(f"Evaluated:           {binary['evaluated']}")
    print(f"Correct:             {binary['correct']}")
    print(f"Missing truth:       {binary['missing_truth']}")
    print(f"Parse failures:      {binary['parse_failures']}")
    print(f"Coverage:            {_format_metric(binary['coverage'])}")
    print(f"Binary accuracy:     {_format_metric(binary['accuracy'])}")
    print(f"End-to-end accuracy: {_format_metric(binary['end_to_end_accuracy'])}")
    print(f"Unsafe precision:    {_format_metric(binary['unsafe_precision'])}")
    print(f"Unsafe recall:       {_format_metric(binary['unsafe_recall'])}")
    print(f"Unsafe F1:           {_format_metric(binary['unsafe_f1'])}")
    print(
        "Confusion (unsafe positive): "
        f"TP={confusion['tp']} FP={confusion['fp']} "
        f"TN={confusion['tn']} FN={confusion['fn']}"
    )

    _print_groups("By dataset", report["by_dataset"])
    _print_groups("By mode", report["by_mode"])

    levels = report["inspecsafe_safety_level"]
    if levels["total_records"]:
        print("\nInspecSafe Level I-IV")
        print("-" * 60)
        print(f"Evaluated:           {levels['evaluated']}")
        print(f"Parse failures:      {levels['parse_failures']}")
        print(f"Accuracy:            {_format_metric(levels['accuracy'])}")
        print(f"End-to-end accuracy: {_format_metric(levels['end_to_end_accuracy'])}")

    print("\nAssociation")
    print("-" * 60)
    for key, value in report["association"].items():
        print(f"{key}: {value}")
    input_stats = report["input"]
    if input_stats.get("invalid_json_lines") or input_stats.get("duplicate_events"):
        print(
            "Input warnings: "
            f"invalid_json_lines={input_stats.get('invalid_json_lines', 0)}, "
            f"duplicate_events={input_stats.get('duplicate_events', 0)}"
        )


def _print_groups(title: str, groups: dict[str, dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * 60)
    for name, metrics in groups.items():
        print(
            f"{name}: accuracy={_format_metric(metrics['accuracy'])} "
            f"evaluated={metrics['evaluated']} "
            f"parse_failures={metrics['parse_failures']}"
        )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local_test_display.py JSONL results."
    )
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable metric report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = evaluate_jsonl(args.results_jsonl)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
    print_report(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nMetrics saved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
