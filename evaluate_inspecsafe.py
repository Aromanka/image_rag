"""Evaluate VLM inference accuracy on the InspecSafe dataset."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROJECT_ROOT, TOP_K, VLM_MAX_NEW_TOKENS
from utils.evaluate_utils import evaluate_results_json, extract_label


def _retrieved_image_paths(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("image_path", ""))
        for item in result.get("retrieved", [])
        if item.get("image_path")
    ]


def run_evaluation(
    dataset_csv: Path,
    mode: str,
    top_k: int,
    max_new_tokens: int,
    limit: int | None,
    offset: int,
) -> None:
    from vlm_inference import VLM_inference, VLM_inference_with_RAG

    df = pd.read_csv(dataset_csv)
    required_cols = {"id", "image_path", "safe_label"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"CSV missing required columns: {required_cols - set(df.columns)}")

    # Slice the dataset
    df = df.iloc[offset:]
    if limit is not None:
        df = df.iloc[:limit]

    if df.empty:
        sys.exit("No samples to evaluate after applying offset/limit.")

    total = len(df)
    errors = 0
    results = []

    print(f"Evaluating {total} samples | mode={mode} | top_k={top_k}")
    print("-" * 60)

    start_time = time.time()

    for idx, row in df.iterrows():
        sample_id = row["id"]
        if hasattr(sample_id, "item"):
            sample_id = sample_id.item()
        image_path = row["image_path"]
        ground_truth = str(row["safe_label"]).strip().lower()

        if ground_truth not in ("safe", "unsafe"):
            print(f"[{sample_id}] SKIP - invalid ground truth: {row['safe_label']}")
            errors += 1
            results.append({
                "id": sample_id,
                "ground_truth": ground_truth,
                "input_image_path": image_path,
                "prompt": None,
                "output": None,
                "error": f"Invalid ground truth: {row['safe_label']}",
            })
            continue

        try:
            if mode == "baseline":
                result = VLM_inference(
                    "safety judgement",
                    image_path,
                    max_new_tokens=max_new_tokens,
                )
            else:
                result = VLM_inference_with_RAG(
                    "safety judgement",
                    image_path,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )

            predicted = extract_label(result["output"])

            if predicted is None:
                status = "PARSE_FAIL"
                errors += 1
            elif predicted == ground_truth:
                status = "CORRECT"
            else:
                status = "WRONG"

            sample_result = {
                "id": sample_id,
                "ground_truth": ground_truth,
                "input_image_path": result.get("query_image", image_path),
                "prompt": result.get("prompt"),
                "output": result.get("output"),
                "predicted": predicted,
                "status": status,
            }
            if mode == "rag":
                sample_result["retrieved_image_paths"] = _retrieved_image_paths(result)
            results.append(sample_result)
            print(f"[{sample_id}] {status} | truth={ground_truth} pred={predicted}")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[{sample_id}] ERROR - {exc}")
            errors += 1
            results.append({
                "id": sample_id,
                "ground_truth": ground_truth,
                "input_image_path": image_path,
                "prompt": None,
                "output": None,
                "predicted": None,
                "status": "ERROR",
                "error": str(exc),
            })

    elapsed = time.time() - start_time
    payload = {
        "metadata": {
            "dataset_csv": str(dataset_csv),
            "mode": mode,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "offset": offset,
            "elapsed_seconds": elapsed,
        },
        "results": results,
    }

    # Save evaluated results to JSON. The same utility can re-evaluate this
    # artifact later if parsing logic changes.
    out_name = f"save/eval_results_{mode}_{int(time.time())}.json"
    out_path = PROJECT_ROOT / out_name
    evaluated_payload = evaluate_results_json(payload, out_path)
    summary = evaluated_payload["summary"]

    print("-" * 60)
    print(f"Total samples:  {summary['total']}")
    print(f"Evaluated:      {summary['evaluated']}")
    print(f"Correct:        {summary['correct']}")
    print(f"Errors/Skipped: {summary['errors_or_skipped']}")
    print(
        "Accuracy:       "
        f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
    )
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"Results saved:  {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM accuracy on InspecSafe dataset."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "InspecSafe" / "dataset.csv",
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag"],
        default="rag",
        help="Inference mode: baseline (no retrieval) or rag (with retrieval).",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-new-tokens", type=int, default=VLM_MAX_NEW_TOKENS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to evaluate. Omit to evaluate all.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of samples to skip from the start.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        dataset_csv=args.dataset_csv,
        mode=args.mode,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        offset=args.offset,
    )

