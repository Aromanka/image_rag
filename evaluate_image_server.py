"""Evaluate the direct backend used by :mod:`image_server`.

This script deliberately bypasses HTTP, response forwarding, and the local-test
display channel.  Inference is dispatched through ``image_server._run_inference``
and parsed through ``image_server._build_success_response_payload`` so mode
routing, prompts, RAG settings, token limits, and response semantics stay in one
place.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

from tqdm import tqdm

import image_server
from config import PROJECT_ROOT, SBERT_MODEL_PATH
from evaluate_constructionsite10k import (
    _assistant_text,
    _sample_image_path as constructionsite_image_path,
    load_constructionsite10k_samples,
)
from evaluate_labsafety_gen import (
    _sample_image_path as labsafety_image_path,
    load_labsafety_gen_samples,
)
from utils.evaluate_utils import parse_constructionsite10k_output


INSPECSAFE = "inspecsafe"
CONSTRUCTIONSITE10K = "constructionsite10k"
LABSAFETY_GEN = "labsafety_gen"
SUPPORTED_DATASETS = {INSPECSAFE, CONSTRUCTIONSITE10K, LABSAFETY_GEN}
ALL_MODES = "all"

DEFAULT_DATASET_PATHS = {
    INSPECSAFE: PROJECT_ROOT / "data" / "inspecsafe" / "test.csv",
    CONSTRUCTIONSITE10K: PROJECT_ROOT / "data" / "constructionsite" / "test.json",
    LABSAFETY_GEN: PROJECT_ROOT / "data" / "lab_safety_gen" / "annotations.jsonl",
}


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    image_path: Path
    ground_truth: str
    reference_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_dataset(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "construction_site_10k": CONSTRUCTIONSITE10K,
        "lab_safety_gen": LABSAFETY_GEN,
        "labsafety": LABSAFETY_GEN,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset {value!r}. Choose one of: {choices}.")
    return normalized


def _resolve_csv_image(raw_value: str, image_root: Path | None) -> Path:
    raw_path = Path(raw_value.strip()).expanduser()
    if image_root is None or (raw_path.is_absolute() and raw_path.is_file()):
        return raw_path

    candidates: list[Path] = []
    normalized = raw_value.replace("\\", "/")
    marker = "/DATA_PATH/"
    if marker in normalized:
        candidates.append(image_root / normalized.split(marker, maxsplit=1)[1])
    if not raw_path.is_absolute():
        candidates.append(image_root / raw_path)
    candidates.append(image_root / raw_path.name)
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_inspecsafe_samples(
    dataset_csv: Path,
    limit: int | None,
    offset: int,
    image_root: Path | None,
) -> list[EvaluationSample]:
    if not dataset_csv.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

    with dataset_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "image_path", "caption", "safe_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"InspecSafe CSV missing columns: {sorted(missing)}")
        rows = list(reader)

    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    samples: list[EvaluationSample] = []
    for row_number, row in enumerate(rows, start=offset + 2):
        truth = str(row["safe_label"]).strip().lower()
        if truth not in {"safe", "unsafe"}:
            raise ValueError(
                f"Invalid safe_label at CSV row {row_number}: {row['safe_label']!r}"
            )
        raw_image = str(row["image_path"])
        samples.append(
            EvaluationSample(
                sample_id=str(row["id"]),
                image_path=_resolve_csv_image(raw_image, image_root),
                ground_truth=truth,
                reference_text=str(row["caption"]).strip(),
                metadata={"source_image": raw_image},
            )
        )
    return samples


def load_constructionsite_samples(
    dataset_json: Path,
    limit: int | None,
    offset: int,
    image_root: Path | None,
) -> list[EvaluationSample]:
    raw_samples = load_constructionsite10k_samples(dataset_json, limit, offset)
    samples: list[EvaluationSample] = []
    for index, raw_sample in enumerate(raw_samples, start=offset):
        image_path = constructionsite_image_path(raw_sample, dataset_json, image_root)
        ground_truth_output = _assistant_text(raw_sample)
        parsed, parse_ok = parse_constructionsite10k_output(ground_truth_output)
        if not parse_ok:
            raise ValueError(
                f"Invalid ConstructionSite-10K ground truth at sample {index}."
            )
        violations = parsed.get("violations", [])
        samples.append(
            EvaluationSample(
                sample_id=image_path.stem,
                image_path=image_path,
                ground_truth="unsafe" if violations else "safe",
                reference_text=str(parsed.get("annotation", "")).strip(),
                metadata={
                    "source_image": raw_sample.get("image", ""),
                    "ground_truth_output": ground_truth_output,
                },
            )
        )
    return samples


def _labsafety_reference_text(sample: dict[str, Any]) -> str:
    hazards = sample.get("hazards", [])
    if isinstance(hazards, list):
        hazard_text = "; ".join(str(item).strip() for item in hazards if str(item).strip())
    else:
        hazard_text = str(hazards).strip()
    # Hazard descriptions are the most relevant reference for unsafe scenes.
    # Safe scenes have no hazards, so use the dataset's scene description.
    return hazard_text or str(sample.get("description", "")).strip()


def load_labsafety_samples(
    annotations_jsonl: Path,
    split: str,
    limit: int | None,
    offset: int,
    image_root: Path | None,
) -> list[EvaluationSample]:
    raw_samples = load_labsafety_gen_samples(annotations_jsonl, split, limit, offset)
    samples: list[EvaluationSample] = []
    label_map = {"hazardous": "unsafe", "non-hazardous": "safe"}
    for index, raw_sample in enumerate(raw_samples, start=offset):
        source_label = str(raw_sample.get("safety_label", "")).strip().lower()
        if source_label not in label_map:
            raise ValueError(
                f"Invalid LabSafety safety_label at sample {index}: {source_label!r}"
            )
        image_path = labsafety_image_path(raw_sample, annotations_jsonl, image_root)
        samples.append(
            EvaluationSample(
                sample_id=str(raw_sample.get("image_id") or image_path.stem),
                image_path=image_path,
                ground_truth=label_map[source_label],
                reference_text=_labsafety_reference_text(raw_sample),
                metadata={
                    "source_image": raw_sample.get("image", ""),
                    "source_label": source_label,
                    "split": raw_sample.get("split", ""),
                },
            )
        )
    return samples


def load_samples(
    *,
    dataset: str,
    dataset_path: Path,
    split: str,
    limit: int | None,
    offset: int,
    image_root: Path | None,
) -> list[EvaluationSample]:
    if dataset == INSPECSAFE:
        return load_inspecsafe_samples(dataset_path, limit, offset, image_root)
    if dataset == CONSTRUCTIONSITE10K:
        return load_constructionsite_samples(dataset_path, limit, offset, image_root)
    return load_labsafety_samples(
        dataset_path, split, limit, offset, image_root
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def calculate_binary_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    successful = [record for record in materialized if record.get("status") == "success"]
    tp = fp = tn = fn = 0
    for record in successful:
        truth = record["ground_truth"]
        predicted = record["server_result"]["safe"]
        if truth == "unsafe" and predicted == "unsafe":
            tp += 1
        elif truth == "safe" and predicted == "unsafe":
            fp += 1
        elif truth == "safe" and predicted == "safe":
            tn += 1
        elif truth == "unsafe" and predicted == "safe":
            fn += 1

    evaluated = tp + fp + tn + fn
    correct = tp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = [float(record["backend_seconds"]) for record in successful]
    sbert_scores = [
        float(record["sbert_score"])
        for record in successful
        if record.get("sbert_score") is not None
    ]
    total = len(materialized)
    return {
        "total": total,
        "evaluated": evaluated,
        "errors": total - len(successful),
        "coverage": evaluated / total if total else 0.0,
        "correct": correct,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "end_to_end_accuracy": correct / total if total else 0.0,
        "sbert_score": statistics.fmean(sbert_scores) if sbert_scores else None,
        "sbert_evaluated": len(sbert_scores),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


class SbertScorer:
    def __init__(self, model_path: Path, device: str, batch_size: int) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"SBERT model directory not found: {model_path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SBERT scoring requires the sentence-transformers package."
            ) from exc
        self.model = SentenceTransformer(str(model_path), device=device)
        self.batch_size = batch_size

    def score_records(self, records: list[dict[str, Any]]) -> None:
        candidates = [
            record
            for record in records
            if record.get("status") == "success" and record.get("reference_text")
        ]
        if not candidates:
            return
        predictions = [str(record["server_result"].get("annotation", "")) for record in candidates]
        references = [str(record["reference_text"]) for record in candidates]
        prediction_embeddings = self.model.encode(
            predictions,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        reference_embeddings = self.model.encode(
            references,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        scores = (prediction_embeddings * reference_embeddings).sum(axis=1)
        for record, score in zip(candidates, scores):
            record["sbert_score"] = float(score)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)


def _evaluate_mode(
    *,
    samples: list[EvaluationSample],
    mode: str,
    top_k: int,
    max_new_tokens: int,
    stage_one_max_new_tokens: int,
    stage_two_max_new_tokens: int,
    checkpoint_every: int,
    checkpoint: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    progress = tqdm(samples, desc=f"mode={mode}", unit="image")
    for sample_index, sample in enumerate(progress, start=1):
        started = time.perf_counter()
        try:
            output, raw_result = image_server._run_inference(
                image_path=sample.image_path,
                mode=mode,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                stage_one_max_new_tokens=stage_one_max_new_tokens,
                stage_two_max_new_tokens=stage_two_max_new_tokens,
            )
            elapsed = time.perf_counter() - started
            server_result = image_server._build_success_response_payload(
                mode, output, raw_result, elapsed
            )
            predicted = str(server_result["safe"])
            record = {
                "id": sample.sample_id,
                "image_path": str(sample.image_path),
                "ground_truth": sample.ground_truth,
                "reference_text": sample.reference_text,
                "status": "success",
                "correct": predicted == sample.ground_truth,
                "backend_seconds": elapsed,
                "sbert_score": None,
                "server_result": server_result,
                "sample_metadata": sample.metadata,
            }
            progress.set_postfix(
                truth=sample.ground_truth,
                pred=predicted,
                seconds=f"{elapsed:.2f}",
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            elapsed = time.perf_counter() - started
            if isinstance(exc, FileNotFoundError):
                tqdm.write(
                    f"[{mode}][{sample.sample_id}] FileNotFoundError: {exc} "
                    f"| query_image={sample.image_path}"
                )
            record = {
                "id": sample.sample_id,
                "image_path": str(sample.image_path),
                "ground_truth": sample.ground_truth,
                "reference_text": sample.reference_text,
                "status": "error",
                "correct": False,
                "backend_seconds": elapsed,
                "sbert_score": None,
                "error": str(exc),
                "sample_metadata": sample.metadata,
            }
            progress.set_postfix(error=type(exc).__name__)
        records.append(record)
        if checkpoint_every and sample_index % checkpoint_every == 0:
            checkpoint(records)
    return records


def _print_summary(mode: str, summary: dict[str, Any]) -> None:
    latency = summary["latency_seconds"]
    sbert = summary["sbert_score"]
    print("-" * 72)
    print(f"Mode:             {mode}")
    print(f"Evaluated:        {summary['evaluated']}/{summary['total']}")
    print(f"Precision:        {summary['precision']:.4f}")
    print(f"Recall:           {summary['recall']:.4f}")
    print(f"F1:               {summary['f1']:.4f}")
    print(f"Accuracy:         {summary['accuracy']:.4f}")
    print(f"End-to-end acc:   {summary['end_to_end_accuracy']:.4f}")
    print(f"SBERT score:      {sbert:.4f}" if sbert is not None else "SBERT score:      skipped")
    if latency["mean"] is not None:
        print(f"Mean latency:     {latency['mean']:.3f}s")
        print(f"P95 latency:      {latency['p95']:.3f}s")
    print(
        "Confusion:        "
        f"TP={summary['tp']} FP={summary['fp']} "
        f"TN={summary['tn']} FN={summary['fn']}"
    )


def run_evaluation(args: argparse.Namespace) -> Path:
    dataset = _normalize_dataset(args.dataset)
    dataset_path = args.dataset_path or DEFAULT_DATASET_PATHS[dataset]
    modes = sorted(image_server.SUPPORTED_MODES) if args.mode == ALL_MODES else [
        image_server._normalize_mode(args.mode)
    ]
    samples = load_samples(
        dataset=dataset,
        dataset_path=dataset_path,
        split=args.split,
        limit=args.limit,
        offset=args.offset,
        image_root=args.image_root,
    )
    if not samples:
        raise ValueError("No samples to evaluate after applying split/offset/limit.")

    output_path = args.output or (
        PROJECT_ROOT
        / "save"
        / f"image_server_eval_{dataset}_{int(time.time())}.json"
    )
    if not args.skip_sbert and not args.sbert_path.is_dir():
        raise FileNotFoundError(f"SBERT model directory not found: {args.sbert_path}")

    if args.lora_weights is not None:
        image_server.configure_lora_weights(args.lora_weights)
    if not args.no_preload:
        print("Preloading the same models used by image_server.py...", flush=True)
        image_server.preload_models()

    report: dict[str, Any] = {
        "metadata": {
            "dataset": dataset,
            "dataset_path": str(dataset_path),
            "split": args.split,
            "modes": modes,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "stage_one_max_new_tokens": args.stage_one_max_new_tokens,
            "stage_two_max_new_tokens": args.stage_two_max_new_tokens,
            "accuracy_gate": image_server.ACCURACY_GATE,
            "rag_dataset": image_server.INSPECSAFE_DATASET,
            "lora_weights": image_server.active_lora_weights(),
            "sbert_model_path": None if args.skip_sbert else str(args.sbert_path),
            "sbert_reference": (
                "server_result.annotation vs dataset reference_text"
            ),
            "positive_label": "unsafe",
            "limit": args.limit,
            "offset": args.offset,
            "created_at_unix": time.time(),
        },
        "summary_by_mode": {},
        "results_by_mode": {},
    }

    scorer: SbertScorer | None = None
    evaluation_started = time.perf_counter()
    for mode in modes:
        def checkpoint(records: list[dict[str, Any]]) -> None:
            report["results_by_mode"][mode] = records
            report["summary_by_mode"][mode] = calculate_binary_metrics(records)
            _write_report(output_path, report)

        records = _evaluate_mode(
            samples=samples,
            mode=mode,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            stage_one_max_new_tokens=args.stage_one_max_new_tokens,
            stage_two_max_new_tokens=args.stage_two_max_new_tokens,
            checkpoint_every=args.checkpoint_every,
            checkpoint=checkpoint,
        )
        # Persist the complete inference output before optional SBERT loading,
        # which can fail independently because it uses a second local model.
        checkpoint(records)
        if not args.skip_sbert:
            if scorer is None:
                print("Loading SBERT scorer...", flush=True)
                scorer = SbertScorer(args.sbert_path, args.sbert_device, args.sbert_batch_size)
            scorer.score_records(records)

        report["results_by_mode"][mode] = records
        summary = calculate_binary_metrics(records)
        report["summary_by_mode"][mode] = summary
        _write_report(output_path, report)
        _print_summary(mode, summary)

    report["metadata"]["elapsed_seconds"] = time.perf_counter() - evaluation_started
    _write_report(output_path, report)
    print(f"Results saved:    {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directly evaluate image_server.py modes without HTTP or the glasses "
            "display channel."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="inspecsafe, constructionsite10k, or labsafety_gen",
    )
    parser.add_argument(
        "--dataset-path",
        "--dataset-csv",
        "--dataset-json",
        "--annotations-jsonl",
        dest="dataset_path",
        type=Path,
        default=None,
        help="Dataset annotation path; defaults depend on --dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=[ALL_MODES, *sorted(image_server.SUPPORTED_MODES), *sorted(image_server.MODE_ALIASES)],
        default=ALL_MODES,
        help="Server mode to evaluate; default evaluates all four modes.",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=image_server.TOP_K)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=image_server.INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--stage-one-max-new-tokens",
        type=int,
        default=image_server.INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--stage-two-max-new-tokens",
        type=int,
        default=image_server.INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save partial results every N images; use 0 to disable.",
    )
    parser.add_argument("--sbert-path", type=Path, default=Path(SBERT_MODEL_PATH))
    parser.add_argument("--sbert-device", default="cpu")
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Match image_server.py --no-preload and load models on first use.",
    )
    image_server.add_lora_cli_arg(parser)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.offset < 0:
        parser.error("--offset cannot be negative.")
    if not 1 <= args.top_k <= image_server.MAX_TOP_K:
        parser.error(f"--top-k must be between 1 and {image_server.MAX_TOP_K}.")
    for name in (
        "max_new_tokens",
        "stage_one_max_new_tokens",
        "stage_two_max_new_tokens",
        "sbert_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative.")
    return args


def main() -> None:
    try:
        run_evaluation(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
