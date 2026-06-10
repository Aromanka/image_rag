"""Evaluate a saved result JSON and export selected RAG sample details."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from utils.evaluate_utils import (
        evaluate_constructionsite10k_results_json,
        evaluate_results_json,
    )
except ModuleNotFoundError:
    from evaluate_utils import evaluate_constructionsite10k_results_json, evaluate_results_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_DIR = PROJECT_ROOT / "demo" / "rag_details"


def _safe_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return name.strip("._") or "sample"


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _copy_image(source: Path | None, destination: Path) -> bool:
    if source is None or not source.is_file():
        print(f"WARNING: image not found, skipped: {source}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, indent=2, ensure_ascii=False, default=str)


def _load_sample_ids(values: list[str] | None, id_file: Path | None) -> set[str]:
    sample_ids: set[str] = set()

    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                sample_ids.add(item)

    if id_file is not None:
        for line in id_file.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                sample_ids.add(item)

    return sample_ids


def _selected_samples(
    results: list[dict[str, Any]],
    sample_ids: set[str],
    include_all: bool,
) -> list[dict[str, Any]]:
    if include_all:
        return results
    return [sample for sample in results if str(sample.get("id")) in sample_ids]


def _detect_dataset_type(results_json: Path) -> str:
    with results_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    results = payload.get("results", []) if isinstance(payload, dict) else payload
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict) and "ground_truth_output" in first:
            return "constructionsite10k"
    return "inspecsafe"


def export_sample_details(
    payload: dict[str, Any],
    sample_ids: set[str],
    demo_dir: str | Path = DEFAULT_DEMO_DIR,
    include_all: bool = False,
) -> int:
    """Copy selected query/retrieved images and prompts into demo folders."""
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Evaluated JSON must contain a 'results' list.")

    target_dir = Path(demo_dir)
    samples = _selected_samples(results, sample_ids, include_all)
    if not samples:
        print("No matching samples found.")
        return 0

    for sample in samples:
        sample_id = sample.get("id", "sample")
        sample_dir = target_dir / _safe_name(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        query_source = _resolve_path(sample.get("input_image_path"))
        query_suffix = query_source.suffix if query_source is not None else ""
        _copy_image(query_source, sample_dir / f"query_image{query_suffix}")

        retrieved_paths = sample.get("retrieved_image_paths") or []
        for index, retrieved_path in enumerate(retrieved_paths, start=1):
            retrieved_source = _resolve_path(retrieved_path)
            suffix = retrieved_source.suffix if retrieved_source is not None else ""
            _copy_image(
                retrieved_source,
                sample_dir / f"retrieved_{index:02d}{suffix}",
            )

        prompt_path = sample_dir / "prompt.txt"
        prompt_path.write_text(_prompt_to_text(sample.get("prompt")), encoding="utf-8")

    return len(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved eval_results JSON and export selected RAG sample "
            "images/prompts under demo/rag_details."
        )
    )
    parser.add_argument("results_json", type=Path, help="Path to eval_results JSON.")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Target sample ids. Supports spaces and comma-separated values.",
    )
    parser.add_argument(
        "--sample-id-file",
        type=Path,
        default=None,
        help="Optional text file containing one target sample id per line.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all samples from the JSON result file.",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=DEFAULT_DEMO_DIR,
        help="Output demo directory for sample detail folders.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional evaluated JSON path. Defaults to overwriting input JSON.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "inspecsafe", "constructionsite10k"],
        default="auto",
        help="Evaluation metric type for the saved JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = _load_sample_ids(args.sample_ids, args.sample_id_file)

    if not args.all and not sample_ids:
        sys.exit("Provide --sample-ids, --sample-id-file, or --all.")

    output_json = args.output_json if args.output_json is not None else args.results_json
    dataset_type = (
        _detect_dataset_type(args.results_json)
        if args.dataset_type == "auto"
        else args.dataset_type
    )
    if dataset_type == "constructionsite10k":
        payload = evaluate_constructionsite10k_results_json(args.results_json, output_json)
    else:
        payload = evaluate_results_json(args.results_json, output_json)
    exported = export_sample_details(
        payload,
        sample_ids=sample_ids,
        demo_dir=args.demo_dir,
        include_all=args.all,
    )

    summary = payload["summary"]
    if dataset_type == "constructionsite10k":
        print(f"Total samples:  {summary['total_samples']}")
        print(f"Valid samples:  {summary['valid_samples']}")
        print(f"Parse failures: {summary['parse_failures']}")
        print(f"Exact match:    {summary['exact_match_acc']:.4f}")
        print(f"Safe/unsafe:    {summary['safe_unsafe_acc']:.4f}")
        print(f"Macro F1:       {summary['macro_f1']:.4f}")
    else:
        print(f"Total samples:  {summary['total']}")
        print(f"Evaluated:      {summary['evaluated']}")
        print(f"Correct:        {summary['correct']}")
        print(f"Errors/Skipped: {summary['errors_or_skipped']}")
        print(
            "Accuracy:       "
            f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
        )
    print(f"Exported:       {exported}")
    print(f"Demo dir:       {Path(args.demo_dir).resolve()}")


if __name__ == "__main__":
    main()
