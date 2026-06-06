"""Evaluate VLM inference accuracy on the InspecSafe dataset."""

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, TOP_K, VLM_MAX_NEW_TOKENS


def extract_label(output: str) -> str | None:
    """Extract safe/unsafe label from VLM output text.

    Looks for 'Final label: safe' or 'Final label: unsafe' first,
    then falls back to the last occurrence of 'safe' or 'unsafe'.
    """
    text = output.strip().lower()

    # Primary: match "Final label: safe/unsafe"
    match = re.search(r"final\s+label\s*:\s*(safe|unsafe)", text)
    if match:
        return match.group(1)

    # Fallback: last standalone occurrence of safe/unsafe
    matches = re.findall(r"\b(unsafe|safe)\b", text)
    if matches:
        return matches[-1]

    return None


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
    correct = 0
    errors = 0
    results = []

    print(f"Evaluating {total} samples | mode={mode} | top_k={top_k}")
    print("-" * 60)

    start_time = time.time()

    for idx, row in df.iterrows():
        sample_id = row["id"]
        image_path = row["image_path"]
        ground_truth = str(row["safe_label"]).strip().lower()

        if ground_truth not in ("safe", "unsafe"):
            print(f"[{sample_id}] SKIP - invalid ground truth: {row['safe_label']}")
            errors += 1
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
                correct += 1
            else:
                status = "WRONG"

            results.append({
                "id": sample_id,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "status": status,
            })
            print(f"[{sample_id}] {status} | truth={ground_truth} pred={predicted}")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[{sample_id}] ERROR - {exc}")
            errors += 1
            results.append({
                "id": sample_id,
                "ground_truth": ground_truth,
                "predicted": None,
                "status": "ERROR",
            })

    elapsed = time.time() - start_time
    evaluated = total - errors
    accuracy = correct / evaluated if evaluated > 0 else 0.0

    print("-" * 60)
    print(f"Total samples:  {total}")
    print(f"Evaluated:      {evaluated}")
    print(f"Correct:        {correct}")
    print(f"Errors/Skipped: {errors}")
    print(f"Accuracy:       {accuracy:.4f} ({correct}/{evaluated})")
    print(f"Time elapsed:   {elapsed:.1f}s")

    # Save results to CSV
    out_name = f"eval_results_{mode}_{int(time.time())}.csv"
    out_path = PROJECT_ROOT / out_name
    pd.DataFrame(results).to_csv(out_path, index=False)
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
