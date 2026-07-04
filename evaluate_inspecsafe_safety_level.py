"""Evaluate baseline or image-RAG inference on InspecSafe safety levels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from config import (
    GATED_RAG,
    INSPECSAFE_DATA_ROOT,
    INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS,
    PROJECT_ROOT,
    SAFETY_LEVEL_TASK,
    SBERT_MODEL_PATH,
    TOP_K,
)
from utils.evaluate_utils import (
    INSPECSAFE_SAFETY_LEVELS,
    evaluate_inspecsafe_safety_level_results_json,
    extract_inspecsafe_safety_level_json,
    normalize_inspecsafe_safety_level,
)
from utils.inspecsafe_paths import pipeline_image_to_dataset_path


def load_inspecsafe_safety_level_data(dataset_json: Path) -> list[dict[str, Any]]:
    """Load the reference pipeline JSON format used by InspecSafe."""
    if not dataset_json.is_file():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")
    with dataset_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("InspecSafe safety-level dataset JSON must contain a list.")
    if not all(isinstance(sample, dict) for sample in payload):
        raise ValueError("Every InspecSafe safety-level sample must be an object.")
    return payload


def assistant_label(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Read the JSON label from the sample's assistant message."""
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        return extract_inspecsafe_safety_level_json(message.get("content"))
    return None


def user_query(sample: dict[str, Any]) -> str | None:
    """Read the textual instruction from a pipeline-format user message."""
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        return text
    return None


def resolve_sample_image(
    sample: dict[str, Any],
    dataset_json: Path,
    image_root: Path | None,
    data_root: Path | None = None,
) -> Path:
    """Resolve a sample against the real dataset or a legacy flat image root."""
    stored_image = str(sample.get("image", "")).strip()
    if not stored_image:
        raise ValueError("Sample is missing the 'image' field.")

    if data_root is not None and image_root is None:
        return pipeline_image_to_dataset_path(stored_image, data_root)

    # Retain compatibility with the reference evaluator's flat pipeline_images
    # directory when --image-root is explicitly provided.
    if image_root is not None:
        return image_root / Path(stored_image).name

    image_path = Path(stored_image).expanduser()
    return image_path if image_path.is_absolute() else dataset_json.parent / image_path


def _retrieved_image_paths(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("image_path", ""))
        for item in result.get("retrieved", [])
        if item.get("image_path")
    ]


def print_safety_level_results(summary: dict[str, Any]) -> None:
    """Print the same metric groups as the provided reference utility."""
    print("\n" + "=" * 62)
    print("InspecSafe Safety Level Evaluation")
    print("=" * 62)
    print(f"  samples          : {summary['n_samples']}")
    print(f"  JSON parse rate  : {summary['json_parse_rate'] * 100:.1f}%")
    print(f"  level accuracy   : {summary['level_accuracy'] * 100:.1f}%")

    print("\n  [Safety Level Classification]")
    print(f"  {'Level':<10} {'n':>5} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-' * 47}")
    for level in INSPECSAFE_SAFETY_LEVELS:
        metrics = summary["per_level"][level]
        print(
            f"  {level:<10} {metrics['n']:>5} "
            f"{metrics['acc'] * 100:>6.1f}% "
            f"{metrics['precision'] * 100:>6.1f}% "
            f"{metrics['recall'] * 100:>6.1f}% "
            f"{metrics['f1'] * 100:>6.1f}%"
        )
    print(f"  {'-' * 47}")
    print(
        f"  {'Macro':<10} {'':>5} {'':>7} "
        f"{summary['level_macro_p'] * 100:>6.1f}% "
        f"{summary['level_macro_r'] * 100:>6.1f}% "
        f"{summary['level_macro_f1'] * 100:>6.1f}%"
    )
    print(
        f"  {'Micro':<10} {'':>5} {'':>7} "
        f"{summary['level_micro_p'] * 100:>6.1f}% "
        f"{summary['level_micro_r'] * 100:>6.1f}% "
        f"{summary['level_micro_f1'] * 100:>6.1f}%"
    )

    print("\n  [Hazard Detection]")
    print(f"  precision  : {summary['hazard_precision'] * 100:.1f}%")
    print(f"  recall     : {summary['hazard_recall'] * 100:.1f}%")
    print(f"  F1         : {summary['hazard_f1'] * 100:.1f}%")
    if summary.get("scene_sbert_sim") is not None:
        print("\n  [Scene Description]")
        print(f"  SBERT sim  : {summary['scene_sbert_sim']:.4f}")


def run_evaluation(
    dataset_json: Path,
    image_root: Path | None,
    mode: str,
    top_k: int,
    gated_rag: float,
    max_new_tokens: int,
    limit: int | None,
    offset: int,
    output_json: Path | None,
    compute_scene_metrics: bool,
    sbert_path: Path | None,
    data_root: Path | None = None,
) -> Path:
    from vlm_inference import VLM_inference, VLM_inference_with_RAG
    from tqdm import tqdm

    samples = load_inspecsafe_safety_level_data(dataset_json)
    samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        sys.exit("No samples to evaluate after applying offset/limit.")

    results: list[dict[str, Any]] = []
    start_time = time.time()
    progress = tqdm(samples, total=len(samples))
    for index, sample in enumerate(progress, start=offset):
        stored_image = str(sample.get("image", ""))
        sample_id = sample.get("id") or Path(stored_image).stem or index
        ground_truth = assistant_label(sample)

        if ground_truth is None:
            results.append({
                "id": sample_id,
                "image": stored_image,
                "ground_truth": {},
                "output": None,
                "error": "Missing or invalid assistant JSON label.",
            })
            progress.set_description(f"[{sample_id}] SKIP - invalid truth")
            continue

        ground_truth_level = normalize_inspecsafe_safety_level(
            ground_truth.get("overall_safety_level")
        )
        if ground_truth_level is None:
            results.append({
                "id": sample_id,
                "image": stored_image,
                "ground_truth": ground_truth,
                "output": None,
                "error": "Invalid ground-truth overall_safety_level.",
            })
            progress.set_description(f"[{sample_id}] SKIP - invalid level")
            continue

        image_path = resolve_sample_image(
            sample,
            dataset_json,
            image_root,
            data_root,
        )
        try:
            if mode == "baseline":
                result = VLM_inference(
                    SAFETY_LEVEL_TASK,
                    image_path,
                    query=user_query(sample),
                    max_new_tokens=max_new_tokens,
                )
            else:
                result = VLM_inference_with_RAG(
                    SAFETY_LEVEL_TASK,
                    image_path,
                    query=user_query(sample),
                    top_k=top_k,
                    gated_rag=gated_rag,
                    max_new_tokens=max_new_tokens,
                )

            predicted_json = extract_inspecsafe_safety_level_json(result.get("output"))
            predicted_level = normalize_inspecsafe_safety_level(
                predicted_json.get("overall_safety_level") if predicted_json else None
            )
            status = (
                "PARSE_FAIL"
                if predicted_json is None
                else "CORRECT"
                if predicted_level == ground_truth_level
                else "WRONG"
            )
            record = {
                "id": sample_id,
                "image": stored_image,
                "input_image_path": str(image_path),
                "ground_truth": ground_truth,
                "prompt": result.get("prompt"),
                "output": result.get("output"),
                "predicted": predicted_json,
                "gt_level": ground_truth_level,
                "pred_level": predicted_level,
                "status": status,
            }
            if mode == "rag":
                record["retrieved_image_paths"] = _retrieved_image_paths(result)
                record["retrieved"] = result.get("retrieved", [])
            results.append(record)
            progress.set_description(
                f"[{sample_id}] {status} | truth={ground_truth_level} "
                f"pred={predicted_level}"
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            results.append({
                "id": sample_id,
                "image": stored_image,
                "input_image_path": str(image_path),
                "ground_truth": ground_truth,
                "prompt": None,
                "output": None,
                "error": str(exc),
            })
            progress.set_description(f"[{sample_id}] ERROR - {exc}")

    elapsed = time.time() - start_time
    target = output_json or (
        PROJECT_ROOT
        / "save"
        / f"eval_results_inspecsafe_safety_level_{mode}_{int(time.time())}.json"
    )
    payload = {
        "metadata": {
            "dataset_json": str(dataset_json),
            "image_root": str(image_root) if image_root else None,
            "data_root": str(data_root) if data_root else None,
            "task_type": SAFETY_LEVEL_TASK,
            "mode": mode,
            "top_k": top_k,
            "gated_rag": gated_rag,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "offset": offset,
            "elapsed_seconds": elapsed,
        },
        "results": results,
    }
    evaluated = evaluate_inspecsafe_safety_level_results_json(
        payload,
        target,
        compute_scene_metrics=compute_scene_metrics,
        sbert_path=sbert_path,
    )
    print_safety_level_results(evaluated["summary"])
    print(f"\n  Time elapsed:   {elapsed:.1f}s")
    print(f"  Results saved:  {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate InspecSafe Level I-IV structured safety outputs."
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "inspecsafe_pipeline" / "pipeline_test.json",
        help="Pipeline-format JSON containing image and assistant label messages.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(INSPECSAFE_DATA_ROOT),
        help=(
            "Original InspecSafe DATA_PATH root. Pipeline image paths are "
            "converted into its split/Annotations hierarchy."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Legacy flat pipeline_images directory. When provided, it takes "
            "precedence over --data-root and images are resolved by basename."
        ),
    )
    parser.add_argument("--mode", choices=["baseline", "rag"], default="rag")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gated-rag",
        "--gated_rag",
        dest="gated_rag",
        type=float,
        default=GATED_RAG,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--sbert-path", type=Path, default=Path(SBERT_MODEL_PATH))
    parser.add_argument(
        "--skip-scene-metrics",
        action="store_true",
        help="Skip scene-description SBERT similarity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        dataset_json=args.dataset_json,
        image_root=args.image_root,
        mode=args.mode,
        top_k=args.top_k,
        gated_rag=args.gated_rag,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        offset=args.offset,
        output_json=args.output_json,
        compute_scene_metrics=not args.skip_scene_metrics,
        sbert_path=args.sbert_path,
        data_root=args.data_root,
    )
