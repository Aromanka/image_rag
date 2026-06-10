"""Evaluate Image_RAG on the ConstructionSite-10K test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from config import (
    CONSTRUCTIONSITE10K_TASK,
    DEFAULT_CONSTRUCTIONSITE10K_QUERY,
    PROJECT_ROOT,
    TOP_K,
)
from utils.evaluate_utils import evaluate_constructionsite10k_results_json


def _message_by_role(sample: dict[str, Any], role: str) -> dict[str, Any] | None:
    for message in sample.get("messages", []):
        if message.get("role") == role:
            return message
    return None


def _user_text(sample: dict[str, Any]) -> str:
    message = _message_by_role(sample, "user")
    if not message:
        return DEFAULT_CONSTRUCTIONSITE10K_QUERY

    content = message.get("content", [])
    if isinstance(content, str):
        return content.strip() or DEFAULT_CONSTRUCTIONSITE10K_QUERY

    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text:
                return text
    return DEFAULT_CONSTRUCTIONSITE10K_QUERY


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


def load_constructionsite10k_samples(
    dataset_json: Path,
    limit: int | None,
    offset: int,
) -> list[dict[str, Any]]:
    if not dataset_json.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_json}")

    with dataset_json.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list):
        raise ValueError("ConstructionSite-10K JSON must contain a list.")

    samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def run_evaluation(
    dataset_json: Path,
    mode: str,
    top_k: int,
    max_new_tokens: int,
    limit: int | None,
    offset: int,
    image_root: Path | None,
) -> None:
    from vlm_inference import VLM_inference, VLM_inference_with_RAG

    samples = load_constructionsite10k_samples(dataset_json, limit, offset)
    if not samples:
        sys.exit("No samples to evaluate after applying offset/limit.")

    print(f"Evaluating {len(samples)} ConstructionSite-10K samples | mode={mode}")
    print("-" * 60)

    results: list[dict[str, Any]] = []
    errors = 0
    start_time = time.time()

    for sample in samples:
        image_path = _sample_image_path(sample, dataset_json, image_root)
        sample_id = image_path.stem
        query = _user_text(sample)
        ground_truth_output = _assistant_text(sample)

        try:
            if mode == "baseline":
                result = VLM_inference(
                    CONSTRUCTIONSITE10K_TASK,
                    image_path,
                    query=query,
                    max_new_tokens=max_new_tokens,
                )
            else:
                result = VLM_inference_with_RAG(
                    CONSTRUCTIONSITE10K_TASK,
                    image_path,
                    query=query,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )

            sample_result = {
                "id": sample_id,
                "input_image_path": result.get("query_image", str(image_path)),
                "ground_truth_output": ground_truth_output,
                "prompt": result.get("prompt"),
                "output": result.get("output"),
            }
            if mode == "rag":
                sample_result["retrieved_image_paths"] = _retrieved_image_paths(result)
            results.append(sample_result)
            print(f"[{sample_id}] inference complete")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            errors += 1
            print(f"[{sample_id}] ERROR - {exc}")
            results.append({
                "id": sample_id,
                "input_image_path": str(image_path),
                "ground_truth_output": ground_truth_output,
                "prompt": None,
                "output": None,
                "error": str(exc),
            })

    elapsed = time.time() - start_time
    payload = {
        "metadata": {
            "dataset_json": str(dataset_json),
            "mode": mode,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "offset": offset,
            "image_root": str(image_root) if image_root else None,
            "elapsed_seconds": elapsed,
            "inference_errors": errors,
        },
        "results": results,
    }

    out_name = f"save/eval_results_constructionsite10k_{mode}_{int(time.time())}.json"
    out_path = PROJECT_ROOT / out_name
    evaluated_payload = evaluate_constructionsite10k_results_json(payload, out_path)
    summary = evaluated_payload["summary"]

    print("-" * 60)
    print(f"Total samples:      {summary['total_samples']}")
    print(f"Valid samples:      {summary['valid_samples']}")
    print(
        "Parse failures:     "
        f"{summary['parse_failures']} ({summary['parse_failure_rate'] * 100:.1f}%)"
    )
    print(f"Exact match:        {summary['exact_match_acc']:.4f}")
    print(f"Safe/unsafe:        {summary['safe_unsafe_acc']:.4f}")
    print(f"Macro F1:           {summary['macro_f1']:.4f}")
    print(f"Micro F1:           {summary['micro_f1']:.4f}")
    print(f"Time elapsed:       {elapsed:.1f}s")
    print(f"Results saved:      {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Image_RAG on ConstructionSite-10K."
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=PROJECT_ROOT / "constructionsite_10k" / "test.json",
        help="Path to ConstructionSite-10K test.json.",
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
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        dataset_json=args.dataset_json,
        mode=args.mode,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        offset=args.offset,
        image_root=args.image_root,
    )
