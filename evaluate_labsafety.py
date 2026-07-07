"""Evaluate Image_RAG on the Lab Safety multiple-choice test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import (
    GATED_RAG,
    DEFAULT_LAB_SAFETY_QUERY,
    LAB_SAFETY_TASK,
    PROJECT_ROOT,
    TOP_K,
    VLM_MAX_NEW_TOKENS,
)
from utils.evaluate_utils import evaluate_labsafety_results_json, extract_choice_label


def _message_by_role(sample: dict[str, Any], role: str) -> dict[str, Any] | None:
    for message in sample.get("messages", []):
        if message.get("role") == role:
            return message
    return None


def _user_text(sample: dict[str, Any]) -> str:
    message = _message_by_role(sample, "user")
    if not message:
        return DEFAULT_LAB_SAFETY_QUERY

    content = message.get("content", [])
    if isinstance(content, str):
        return content.strip() or DEFAULT_LAB_SAFETY_QUERY

    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text:
                return text
    return DEFAULT_LAB_SAFETY_QUERY


def _assistant_text(sample: dict[str, Any]) -> str:
    message = _message_by_role(sample, "assistant")
    if not message:
        return ""
    return str(message.get("content", "")).strip()


def _sample_image_path(
    sample: dict[str, Any],
    dataset_json: Path,
    image_root: Path | None,
) -> Path:
    raw_image = str(sample.get("image", "")).replace("\\", "/")
    image_path = Path(raw_image)
    if image_path.is_absolute():
        return image_path

    if image_root is not None:
        return image_root / image_path.name
    return dataset_json.parent / image_path


def _retrieved_image_paths(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("image_path", ""))
        for item in result.get("retrieved", [])
        if item.get("image_path")
    ]


def load_labsafety_samples(
    dataset_json: Path,
    limit: int | None,
    offset: int,
) -> list[dict[str, Any]]:
    if not dataset_json.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_json}")

    with dataset_json.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list):
        raise ValueError("Lab Safety JSON must contain a list.")

    samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def _metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def run_evaluation(
    dataset_json: Path,
    mode: str,
    top_k: int,
    max_new_tokens: int,
    limit: int | None,
    offset: int,
    image_root: Path | None,
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

    samples = load_labsafety_samples(dataset_json, limit, offset)
    if not samples:
        sys.exit("No samples to evaluate after applying offset/limit.")

    print(
        f"Evaluating {len(samples)} Lab Safety samples | mode={mode} "
        f"| top_k={top_k} | gated_rag={gated_rag}"
    )
    print("-" * 60)

    results: list[dict[str, Any]] = []
    errors = 0
    start_time = time.time()
    total = len(samples)
    pbar = tqdm(list(range(total)))

    for sample_idx in pbar:
        sample = samples[sample_idx]
        metadata = _metadata(sample)
        image_path = _sample_image_path(sample, dataset_json, image_root)
        sample_id = image_path.stem
        query = _user_text(sample)
        ground_truth = str(metadata.get("answer") or _assistant_text(sample)).strip().upper()

        try:
            if mode == "baseline":
                result = VLM_inference(
                    LAB_SAFETY_TASK,
                    image_path,
                    query=query,
                    max_new_tokens=max_new_tokens,
                )
            else:
                result = VLM_inference_with_RAG(
                    LAB_SAFETY_TASK,
                    image_path,
                    query=query,
                    top_k=top_k,
                    gated_rag=gated_rag,
                    max_new_tokens=max_new_tokens,
                )

            predicted = extract_choice_label(result.get("output"))
            if predicted is None:
                status = "PARSE_FAIL"
                errors += 1
            elif predicted == ground_truth:
                status = "CORRECT"
            else:
                status = "WRONG"

            sample_result = {
                "id": sample_id,
                "question": query,
                "ground_truth_answer": ground_truth,
                "ground_truth_explanation": metadata.get("explanation", ""),
                "category": metadata.get("category", []),
                "level": metadata.get("level", ""),
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
                "question": query,
                "ground_truth_answer": ground_truth,
                "ground_truth_explanation": metadata.get("explanation", ""),
                "category": metadata.get("category", []),
                "level": metadata.get("level", ""),
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
            "dataset_json": str(dataset_json),
            "mode": mode,
            "top_k": top_k,
            "gated_rag": gated_rag,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "offset": offset,
            "image_root": str(image_root) if image_root else None,
            "lora_weights": active_lora_weights(),
            "elapsed_seconds": elapsed,
            "inference_errors": errors,
        },
        "results": results,
    }

    out_name = f"save/eval_results_labsafety_{mode}_{int(time.time())}.json"
    out_path = PROJECT_ROOT / out_name
    evaluated_payload = evaluate_labsafety_results_json(payload, out_path)
    summary = evaluated_payload["summary"]

    print("-" * 60)
    print(f"Total samples:  {summary['total']}")
    print(f"Evaluated:      {summary['evaluated']}")
    print(f"Correct:        {summary['correct']}")
    print(f"Errors/Skipped: {summary['errors_or_skipped']}")
    print(f"Parse failures: {summary['parse_failures']}")
    print(
        "Accuracy:       "
        f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
    )
    print("Confusion matrix rows=truth, columns=prediction:")
    labels = ["A", "B", "C", "D", "PARSE_FAIL"]
    print("Truth/Pred      " + " ".join(f"{label:>10}" for label in labels))
    for truth in ["A", "B", "C", "D"]:
        row = evaluated_payload["confusion"][truth]
        print(f"{truth:<15}" + " ".join(f"{row[label]:>10}" for label in labels))
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"Results saved:  {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Image_RAG on Lab Safety multiple-choice JSON."
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "lab_safety" / "lab_test.json",
        help="Path to Lab Safety test JSON.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional image root. Defaults to resolving paths relative to dataset JSON.",
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
    from vlm_inference import add_lora_cli_arg

    add_lora_cli_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        dataset_json=args.dataset_json,
        mode=args.mode,
        top_k=args.top_k,
        gated_rag=args.gated_rag,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        offset=args.offset,
        image_root=args.image_root,
        lora_weights=args.lora_weights,
    )
