"""Export self-contained Image_RAG dataset subsets for a display computer."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INSPECSAFE_DATA_ROOT  # noqa: E402
from utils.local_test_data import (  # noqa: E402
    INSPECSAFE_SAFETY_LEVEL_DATASET,
    LABSAFETY_GEN_DATASET,
    DisplaySample,
    default_annotations_path,
    load_display_samples,
)


EXPORT_FORMAT_VERSION = 1
DATASET_CHOICES = (
    INSPECSAFE_SAFETY_LEVEL_DATASET,
    LABSAFETY_GEN_DATASET,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def select_samples(
    samples: list[DisplaySample],
    *,
    requested_ids: list[str],
    offset: int,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[DisplaySample]:
    """Select exact IDs in requested order, or a reproducible positional subset."""
    if requested_ids:
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("Requested sample IDs must not contain duplicates.")
        by_id: dict[str, DisplaySample] = {}
        duplicate_ids: set[str] = set()
        for sample in samples:
            if sample.sample_id in by_id:
                duplicate_ids.add(sample.sample_id)
            by_id[sample.sample_id] = sample
        ambiguous = [sample_id for sample_id in requested_ids if sample_id in duplicate_ids]
        if ambiguous:
            raise ValueError(
                "Sample IDs are not unique in the source dataset: "
                + ", ".join(ambiguous)
            )
        missing = [sample_id for sample_id in requested_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                "Requested sample IDs were not found among displayable source images: "
                + ", ".join(missing)
            )
        return [by_id[sample_id] for sample_id in requested_ids]

    selected = list(samples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def export_dataset(
    *,
    dataset: str,
    samples: list[DisplaySample],
    destination: Path,
    source_annotations: Path,
) -> dict[str, Any]:
    """Copy images, emit compatible annotations, and return manifest metadata."""
    destination.mkdir(parents=True, exist_ok=False)
    exported_annotations: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        split = _sample_split(sample)
        suffix = sample.image_path.suffix.lower() or ".img"
        filename = f"{index:06d}_{_safe_filename(sample.sample_id)}{suffix}"
        relative_image = Path("images") / split / filename
        target_image = destination / relative_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, target_image)
        digest = _sha256_file(target_image)

        if dataset == INSPECSAFE_SAFETY_LEVEL_DATASET:
            exported_annotations.append(
                _inspecsafe_annotation(sample, relative_image, split)
            )
        else:
            exported_annotations.append(
                _labsafety_annotation(sample, relative_image, split)
            )
        manifest_items.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "source_image": str(sample.image_path.resolve()),
                "exported_image": relative_image.as_posix(),
                "sha256": digest,
                "size_bytes": target_image.stat().st_size,
            }
        )

    if dataset == INSPECSAFE_SAFETY_LEVEL_DATASET:
        annotation_path = destination / "annotations.json"
        annotation_path.write_text(
            json.dumps(exported_annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        annotation_path = destination / "annotations.jsonl"
        with annotation_path.open("w", encoding="utf-8", newline="\n") as file:
            for annotation in exported_annotations:
                file.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    _verify_export(
        dataset=dataset,
        annotations_path=annotation_path,
        image_root=destination,
        expected_count=len(samples),
    )
    return {
        "dataset": dataset,
        "count": len(samples),
        "source_annotations": str(source_annotations.resolve()),
        "annotations": annotation_path.relative_to(destination.parent).as_posix(),
        "image_root": destination.relative_to(destination.parent).as_posix(),
        "items": manifest_items,
    }


def export_subsets(
    *,
    output_dir: Path,
    datasets: list[str],
    annotations_by_dataset: dict[str, Path],
    image_roots_by_dataset: dict[str, Path | None],
    ids_by_dataset: dict[str, list[str]],
    split: str,
    offset: int,
    limit: int | None,
    shuffle: bool,
    seed: int,
    strict_images: bool,
) -> dict[str, Any]:
    """Build the complete portable data directory atomically."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new directory."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.tmp-{uuid4().hex[:8]}"
    )
    staging_dir.mkdir(parents=False, exist_ok=False)

    manifest: dict[str, Any] = {
        "format": "image_rag_local_test_data",
        "format_version": EXPORT_FORMAT_VERSION,
        "created_at": utc_now_iso(),
        "selection": {
            "split": split,
            "offset": offset,
            "limit_per_dataset": limit,
            "shuffle": shuffle,
            "seed": seed,
            "strict_images": strict_images,
        },
        "datasets": {},
    }
    try:
        for dataset in datasets:
            annotations_path = annotations_by_dataset[dataset].expanduser().resolve()
            image_root = image_roots_by_dataset[dataset]
            if image_root is not None:
                image_root = image_root.expanduser().resolve()
            available, missing_count = load_display_samples(
                dataset=dataset,
                annotations_path=annotations_path,
                image_root=image_root,
                split=split,
                skip_missing=not strict_images,
                sample_ids=(
                    set(ids_by_dataset.get(dataset, []))
                    if ids_by_dataset.get(dataset)
                    else None
                ),
            )
            selected = select_samples(
                available,
                requested_ids=ids_by_dataset.get(dataset, []),
                offset=offset,
                limit=limit,
                shuffle=shuffle,
                seed=seed,
            )
            if not selected:
                raise ValueError(
                    f"No displayable {dataset} samples matched the selection. "
                    "Check the annotations, image root, split, IDs, offset, and limit."
                )
            dataset_manifest = export_dataset(
                dataset=dataset,
                samples=selected,
                destination=staging_dir / dataset,
                source_annotations=annotations_path,
            )
            dataset_manifest["missing_source_images_skipped"] = missing_count
            requested_ids = ids_by_dataset.get(dataset, [])
            dataset_manifest["selection"] = {
                "mode": "exact_ids" if requested_ids else "positional",
                "requested_ids": requested_ids,
                "eligible_source_images": len(available),
            }
            manifest["datasets"][dataset] = dataset_manifest

        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return manifest


def _inspecsafe_annotation(
    sample: DisplaySample,
    relative_image: Path,
    split: str,
) -> dict[str, Any]:
    if sample.source_annotation:
        annotation = copy.deepcopy(sample.source_annotation)
        annotation.setdefault("id", sample.sample_id)
        annotation["image"] = relative_image.as_posix()
        metadata = annotation.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            annotation["metadata"] = metadata
        metadata.setdefault("split", split)
        return annotation

    metadata = dict(sample.metadata)
    metadata.setdefault("split", split)
    return {
        "id": sample.sample_id,
        "image": relative_image.as_posix(),
        "messages": [
            {
                "role": "assistant",
                "content": json.dumps(sample.ground_truth, ensure_ascii=False),
            }
        ],
        "metadata": metadata,
    }


def _labsafety_annotation(
    sample: DisplaySample,
    relative_image: Path,
    split: str,
) -> dict[str, Any]:
    annotation = copy.deepcopy(sample.source_annotation)
    annotation.update(
        {
            "image_id": sample.sample_id,
            "image": relative_image.as_posix(),
            "split": split,
            "safety_label": sample.ground_truth.get("safety_label"),
            "hazards": sample.ground_truth.get("hazards", []),
            "description": sample.ground_truth.get("description", ""),
            "vlm_label": sample.metadata.get("vlm_label"),
            "agree": sample.metadata.get("agree"),
        }
    )
    return annotation


def _sample_split(sample: DisplaySample) -> str:
    split = str(sample.metadata.get("split", "")).strip().lower()
    if split in {"train", "test"}:
        return split
    normalized_source = sample.source_image.replace("\\", "/")
    if "/train/" in f"/{normalized_source}":
        return "train"
    if "/test/" in f"/{normalized_source}":
        return "test"
    basename = normalized_source.rsplit("/", maxsplit=1)[-1]
    prefix = basename.split("__", maxsplit=1)[0].lower()
    return prefix if prefix in {"train", "test"} else "unspecified"


def _safe_filename(sample_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._")
    return sanitized[:120] or "sample"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_export(
    *,
    dataset: str,
    annotations_path: Path,
    image_root: Path,
    expected_count: int,
) -> None:
    loaded, missing = load_display_samples(
        dataset=dataset,
        annotations_path=annotations_path,
        image_root=image_root,
        split="all",
        skip_missing=False,
    )
    if missing or len(loaded) != expected_count:
        raise RuntimeError(
            f"Export verification failed for {dataset}: "
            f"expected={expected_count}, loaded={len(loaded)}, missing={missing}."
        )


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected InspecSafe safety-level and/or LabSafety-Gen samples "
            "into a portable local-display data directory."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / f"local_test_export_{timestamp}",
        help="New directory to create. Existing directories are never overwritten.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_CHOICES,
        help="Dataset to export; repeat as needed. Defaults to both datasets.",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum samples per selected dataset after offset/shuffle.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict-images",
        action="store_true",
        help="Fail instead of skipping source annotations whose image is missing.",
    )
    parser.add_argument(
        "--inspecsafe-id",
        action="append",
        default=[],
        help="Exact InspecSafe sample ID to export; repeat to preserve this order.",
    )
    parser.add_argument(
        "--labsafety-id",
        action="append",
        default=[],
        help="Exact LabSafety-Gen image_id to export; repeat to preserve this order.",
    )
    parser.add_argument(
        "--inspecsafe-annotations",
        type=Path,
        default=default_annotations_path(
            INSPECSAFE_SAFETY_LEVEL_DATASET,
            PROJECT_ROOT,
        ),
    )
    parser.add_argument(
        "--inspecsafe-image-root",
        type=Path,
        default=Path(INSPECSAFE_DATA_ROOT),
        help="Original InspecSafe DATA_PATH or flat pipeline image directory.",
    )
    parser.add_argument(
        "--labsafety-annotations",
        type=Path,
        default=default_annotations_path(LABSAFETY_GEN_DATASET, PROJECT_ROOT),
    )
    parser.add_argument(
        "--labsafety-image-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "lab_safety_gen",
        help="LabSafety-Gen root containing images/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise SystemExit("--offset cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    datasets = list(dict.fromkeys(args.dataset or DATASET_CHOICES))
    if args.inspecsafe_id and INSPECSAFE_SAFETY_LEVEL_DATASET not in datasets:
        raise SystemExit("--inspecsafe-id requires exporting inspecsafe_safety_level.")
    if args.labsafety_id and LABSAFETY_GEN_DATASET not in datasets:
        raise SystemExit("--labsafety-id requires exporting labsafety_gen.")

    try:
        manifest = export_subsets(
            output_dir=args.output_dir,
            datasets=datasets,
            annotations_by_dataset={
                INSPECSAFE_SAFETY_LEVEL_DATASET: args.inspecsafe_annotations,
                LABSAFETY_GEN_DATASET: args.labsafety_annotations,
            },
            image_roots_by_dataset={
                INSPECSAFE_SAFETY_LEVEL_DATASET: args.inspecsafe_image_root,
                LABSAFETY_GEN_DATASET: args.labsafety_image_root,
            },
            ids_by_dataset={
                INSPECSAFE_SAFETY_LEVEL_DATASET: args.inspecsafe_id,
                LABSAFETY_GEN_DATASET: args.labsafety_id,
            },
            split=args.split,
            offset=args.offset,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
            strict_images=args.strict_images,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Export failed: {exc}") from exc

    print(f"Export complete: {args.output_dir.expanduser().resolve()}")
    for dataset, details in manifest["datasets"].items():
        print(
            f"  {dataset}: {details['count']} samples "
            f"({details['missing_source_images_skipped']} missing source images skipped)"
        )


if __name__ == "__main__":
    main()
