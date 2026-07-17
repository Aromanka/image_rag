"""Dataset adapters for the local-test image display client."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from utils.evaluate_utils import extract_inspecsafe_safety_level_json
from utils.inspecsafe_paths import pipeline_image_to_dataset_path


INSPECSAFE_SAFETY_LEVEL_DATASET = "inspecsafe_safety_level"
LABSAFETY_GEN_DATASET = "labsafety_gen"
SUPPORTED_LOCAL_TEST_DATASETS = {
    INSPECSAFE_SAFETY_LEVEL_DATASET,
    LABSAFETY_GEN_DATASET,
}


@dataclass(frozen=True)
class DisplaySample:
    dataset: str
    sample_id: str
    image_path: Path
    source_image: str
    ground_truth: dict[str, Any]
    metadata: dict[str, Any]
    source_annotation: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def as_record(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "source_image": self.source_image,
            "image_path": str(self.image_path),
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


def default_annotations_path(dataset: str, project_root: Path) -> Path:
    normalized = normalize_dataset(dataset)
    if normalized == INSPECSAFE_SAFETY_LEVEL_DATASET:
        return project_root / "data" / "inspecsafe_pipeline" / "pipeline_test.json"
    return project_root / "data" / "lab_safety_gen" / "annotations.jsonl"


def normalize_dataset(dataset: str) -> str:
    normalized = dataset.strip().lower().replace("-", "_")
    aliases = {
        "inspecsafe": INSPECSAFE_SAFETY_LEVEL_DATASET,
        "inspecsafe_level": INSPECSAFE_SAFETY_LEVEL_DATASET,
        "lab_safety_gen": LABSAFETY_GEN_DATASET,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_LOCAL_TEST_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_LOCAL_TEST_DATASETS))
        raise ValueError(f"Unsupported local-test dataset {dataset!r}: {choices}.")
    return normalized


def load_display_samples(
    *,
    dataset: str,
    annotations_path: Path,
    image_root: Path | None,
    split: str = "test",
    skip_missing: bool = True,
    sample_ids: set[str] | None = None,
) -> tuple[list[DisplaySample], int]:
    """Load display samples and return ``(samples, missing_image_count)``."""
    normalized = normalize_dataset(dataset)
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all.")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    if normalized == INSPECSAFE_SAFETY_LEVEL_DATASET:
        raw_samples = _read_inspecsafe_samples(annotations_path)
        adapter = _adapt_inspecsafe_sample
    else:
        raw_samples = _read_jsonl_objects(annotations_path)
        adapter = _adapt_labsafety_sample

    samples: list[DisplaySample] = []
    missing = 0
    for index, raw_sample in enumerate(raw_samples):
        sample_split = _sample_split(raw_sample, normalized)
        if normalized_split != "all" and sample_split != normalized_split:
            continue
        sample = adapter(raw_sample, index, annotations_path, image_root)
        if sample_ids is not None and sample.sample_id not in sample_ids:
            continue
        if not sample.image_path.is_file():
            missing += 1
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"Image for sample {sample.sample_id!r} not found: "
                f"{sample.image_path}"
            )
        samples.append(sample)
    return samples, missing


def _read_inspecsafe_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("InspecSafe annotations must be a JSON list of objects.")
    return payload


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_number} is not an object.")
            samples.append(item)
    return samples


def _sample_split(sample: dict[str, Any], dataset: str) -> str:
    if dataset == LABSAFETY_GEN_DATASET:
        return str(sample.get("split", "")).strip().lower()
    metadata = sample.get("metadata")
    if isinstance(metadata, dict) and metadata.get("split"):
        return str(metadata["split"]).strip().lower()
    stored_image = str(sample.get("image", "")).replace("\\", "/")
    basename = stored_image.rsplit("/", maxsplit=1)[-1]
    return basename.split("__", maxsplit=1)[0].strip().lower()


def _adapt_inspecsafe_sample(
    sample: dict[str, Any],
    index: int,
    annotations_path: Path,
    image_root: Path | None,
) -> DisplaySample:
    source_image = str(sample.get("image", "")).strip()
    if not source_image:
        raise ValueError(f"InspecSafe sample {index} has no image field.")
    ground_truth = _inspecsafe_ground_truth(sample)
    metadata = sample.get("metadata")
    sample_id = str(sample.get("id") or Path(source_image).stem or index)
    return DisplaySample(
        dataset=INSPECSAFE_SAFETY_LEVEL_DATASET,
        sample_id=sample_id,
        image_path=_resolve_image(
            source_image,
            annotations_path,
            image_root,
            is_inspecsafe=True,
        ),
        source_image=source_image,
        ground_truth=ground_truth,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        source_annotation=dict(sample),
    )


def _inspecsafe_ground_truth(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            parsed = extract_inspecsafe_safety_level_json(message.get("content"))
            if parsed is not None:
                return parsed
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        return {
            "hazards": metadata.get("hazards", []),
            "overall_safety_level": metadata.get("overall_safety_level"),
        }
    return {}


def _adapt_labsafety_sample(
    sample: dict[str, Any],
    index: int,
    annotations_path: Path,
    image_root: Path | None,
) -> DisplaySample:
    source_image = str(sample.get("image", "")).strip()
    if not source_image:
        raise ValueError(f"LabSafety sample {index} has no image field.")
    sample_id = str(sample.get("image_id") or Path(source_image).stem or index)
    ground_truth = {
        "safety_label": sample.get("safety_label"),
        "hazards": sample.get("hazards", []),
        "description": sample.get("description", ""),
    }
    metadata = {
        "split": sample.get("split"),
        "vlm_label": sample.get("vlm_label"),
        "agree": sample.get("agree"),
    }
    return DisplaySample(
        dataset=LABSAFETY_GEN_DATASET,
        sample_id=sample_id,
        image_path=_resolve_image(
            source_image,
            annotations_path,
            image_root,
            is_inspecsafe=False,
        ),
        source_image=source_image,
        ground_truth=ground_truth,
        metadata=metadata,
        source_annotation=dict(sample),
    )


def _resolve_image(
    stored_image: str,
    annotations_path: Path,
    image_root: Path | None,
    *,
    is_inspecsafe: bool,
) -> Path:
    stored_path = Path(stored_image).expanduser()
    if stored_path.is_absolute():
        return stored_path

    candidates: list[Path] = []
    if image_root is not None:
        candidates.append(image_root / stored_path)
        if is_inspecsafe:
            try:
                candidates.append(
                    pipeline_image_to_dataset_path(stored_image, image_root)
                )
            except ValueError:
                pass
        candidates.append(image_root / stored_path.name)
    candidates.append(annotations_path.parent / stored_path)
    return _first_existing_or_first(candidates)


def _first_existing_or_first(candidates: Iterable[Path]) -> Path:
    materialized = list(candidates)
    for candidate in materialized:
        if candidate.is_file():
            return candidate
    if not materialized:
        raise ValueError("No image path candidates were generated.")
    return materialized[0]
