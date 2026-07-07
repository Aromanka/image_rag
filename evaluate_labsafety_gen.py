"""Evaluate Image_RAG on the LabSafety-v1 generated lab-safety test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import (
    DEFAULT_LAB_SAFETY_GEN_QUERY,
    GATED_RAG,
    LAB_SAFETY_GEN_TASK,
    PROJECT_ROOT,
    TOP_K,
    VLM_MAX_NEW_TOKENS,
)
from utils.evaluate_utils import evaluate_labsafety_gen_results_json, extract_hazard_label


def load_labsafety_gen_samples(
    annotations_jsonl: Path,
    split: str,
    limit: int | None,
    offset: int,
) -> list[dict[str, Any]]:
    if not annotations_jsonl.is_file():
        raise FileNotFoundError(f"Dataset not found: {annotations_jsonl}")

    split = split.strip().lower()
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all.")

    samples: list[dict[str, Any]] = []
    with annotations_jsonl.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_split = str(sample.get("split", "")).strip().lower()
            if split == "all" or sample_split == split:
                samples.append(sample)

    samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def _sample_image_path(
    sample: dict[str, Any],
    annotations_jsonl: Path,
    image_root: Path | None,
) -> Path:
    raw_image = str(sample.get("image", "")).replace("\\", "/")
    image_path = Path(raw_image)
    if image_path.is_absolute():
        return image_path

    if image_root is not None:
        rooted_path = image_root / image_path
        if rooted_path.is_file():
            return rooted_path
        return image_root / image_path.name
    return annotations_jsonl.parent / image_path


def _retrieved_image_paths(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("image_path", ""))
        for item in result.get("retrieved", [])
        if item.get("image_path")
    ]


def _hazards_text(sample: dict[str, Any]) -> str:
    hazards = sample.get("hazards", [])
    if isinstance(hazards, list):
        return "; ".join(str(item) for item in hazards)
    return str(hazards)


def run_evaluation(
    annotations_jsonl: Path,
    split: str,
    mode: str,
    top_k: int,
    max_new_tokens: int,
    limit: int | None,
    offset: int,
    image_root: Path | None,
    query: str,
    gated_rag: float = GATED_RAG,
    lora_weights: str | Path | None = None,
) -> None:
    from vlm_inference import (
        VLM_inference,
        VLM_inference_with_RAG,
        active_lora_weights,
        configure_lora_weights,
    )

    if lora_weights is not None:
        configure_lora_weights(lora_weights)

    samples = load_labsafety_gen_samples(annotations_jsonl, split, limit, offset)
    if not samples:
        sys.exit("No samples to evaluate after applying split/offset/limit.")

    print(
        f"Evaluating {len(samples)} LabSafety-v1 samples | split={split} "
        f"| mode={mode} | top_k={top_k} | gated_rag={gated_rag}"
    )
    print("-" * 60)

    results: list[dict[str, Any]] = []
    errors = 0
    start_time = time.time()
    total = len(samples)
    pbar = tqdm(list(range(total)))

    for sample_idx in pbar:
        sample = samples[sample_idx]
        image_path = _sample_image_path(sample, annotations_jsonl, image_root)
        sample_id = str(sample.get("image_id") or image_path.stem)
        ground_truth = str(sample.get("safety_label", "")).strip().lower()

        try:
            if mode == "baseline":
                result = VLM_inference(
                    LAB_SAFETY_GEN_TASK,
                    image_path,
                    query=query,
                    max_new_tokens=max_new_tokens,
                )
            else:
                result = VLM_inference_with_RAG(
                    LAB_SAFETY_GEN_TASK,
                    image_path,
                    query=query,
                    top_k=top_k,
                    gated_rag=gated_rag,
                    max_new_tokens=max_new_tokens,
                )

            predicted = extract_hazard_label(result.get("output"))
            if predicted is None:
                status = "PARSE_FAIL"
                errors += 1
            elif predicted == ground_truth:
                status = "CORRECT"
            else:
                status = "WRONG"

            sample_result = {
                "id": sample_id,
                "ground_truth_hazard_label": ground_truth,
                "hazards": _hazards_text(sample),
                "description": sample.get("description", ""),
                "vlm_label": sample.get("vlm_label", ""),
                "agree": sample.get("agree", ""),
                "split": sample.get("split", ""),
                "input_image_path": result.get("query_image", str(image_path)),
                "prompt": result.get("prompt"),
                "output": result.get("output"),
                "predicted": predicted,
                "status": status,
            }
            if mode == "rag":
                sample_result["retrieved_image_paths"] = _retrieved_image_paths(result)
            results.append(sample_result)
            pbar.set_description(
                f"[{sample_id}] {status} | truth={ground_truth} pred={predicted}"
            )

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            errors += 1
            print(f"[{sample_id}] ERROR - {exc}")
            results.append({
                "id": sample_id,
                "ground_truth_hazard_label": ground_truth,
                "hazards": _hazards_text(sample),
                "description": sample.get("description", ""),
                "vlm_label": sample.get("vlm_label", ""),
                "agree": sample.get("agree", ""),
                "split": sample.get("split", ""),
                "input_image_path": str(image_path),
                "prompt": None,
                "output": None,
                "predicted": None,
                "status": "ERROR",
                "error": str(exc),
            })

        elapsed = time.time() - start_time
        avg_time = elapsed / (sample_idx + 1)
        eta = avg_time * (total - sample_idx - 1)
        pbar.set_postfix({"Avg": f"{avg_time:.2f}s", "ETA": f"{eta:.0f}s"})

    elapsed = time.time() - start_time
    payload = {
        "metadata": {
            "annotations_jsonl": str(annotations_jsonl),
            "split": split,
            "mode": mode,
            "top_k": top_k,
            "gated_rag": gated_rag,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "offset": offset,
            "image_root": str(image_root) if image_root else None,
            "query": query,
            "lora_weights": active_lora_weights(),
            "elapsed_seconds": elapsed,
            "inference_errors": errors,
        },
        "results": results,
    }

    out_name = f"save/eval_results_labsafety_gen_{mode}_{int(time.time())}.json"
    out_path = PROJECT_ROOT / out_name
    evaluated_payload = evaluate_labsafety_gen_results_json(payload, out_path)
    summary = evaluated_payload["summary"]

    print("-" * 60)
    print(f"Total samples:  {summary['total']}")
    print(f"Evaluated:      {summary['evaluated']}")
    print(f"Correct:        {summary['correct']}")
    print(f"Errors/Skipped: {summary['errors_or_skipped']}")
    print(f"Parse failures: {summary['parse_failures']}")
    print(f"TP:             {summary['tp']}")
    print(f"FP:             {summary['fp']}")
    print(f"TN:             {summary['tn']}")
    print(f"FN:             {summary['fn']}")
    print(f"Hazard F1:      {summary['hazardous_f1']:.4f}")
    print(
        "Accuracy:       "
        f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
    )
    print("Confusion matrix rows=truth, columns=prediction:")
    labels = ["hazardous", "non-hazardous", "PARSE_FAIL"]
    print("Truth/Pred      " + " ".join(f"{label:>14}" for label in labels))
    for truth in ["hazardous", "non-hazardous"]:
        row = evaluated_payload["confusion"][truth]
        print(f"{truth:<15}" + " ".join(f"{row[label]:>14}" for label in labels))
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"Results saved:  {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Image_RAG on LabSafety-v1 JSONL."
    )
    parser.add_argument(
        "--annotations-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "lab_safety_gen" / "annotations.jsonl",
        help="Path to LabSafety-v1 annotations.jsonl.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional image root. Defaults to resolving paths relative to JSONL.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag"],
        default="rag",
        help="Inference mode.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gated-rag",
        "--gated_rag",
        dest="gated_rag",
        type=float,
        default=GATED_RAG,
        help="Keep top-k RAG results with cosine similarity >= this threshold.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=VLM_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query", type=str, default=DEFAULT_LAB_SAFETY_GEN_QUERY)
    from vlm_inference import add_lora_cli_arg

    add_lora_cli_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        annotations_jsonl=args.annotations_jsonl,
        split=args.split,
        mode=args.mode,
        top_k=args.top_k,
        gated_rag=args.gated_rag,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        offset=args.offset,
        image_root=args.image_root,
        query=args.query,
        lora_weights=args.lora_weights,
    )
