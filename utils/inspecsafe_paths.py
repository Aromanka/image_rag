"""Path conversion helpers shared by InspecSafe indexing and evaluation."""

from __future__ import annotations

import re
from pathlib import Path

from config import INSPECSAFE_DATA_ROOT


def pipeline_image_to_dataset_path(
    pipeline_image: str,
    data_root: str | Path = INSPECSAFE_DATA_ROOT,
) -> Path:
    """Convert a flattened pipeline image path to the original dataset path.

    Pipeline paths have the form
    ``images/{split}__{instance}__{filename}``. In the original InspecSafe
    tree, Level04 samples are normal and Levels 01-03 are anomalous.
    """
    normalized = str(pipeline_image).strip().replace("\\", "/")
    filename_with_context = normalized.rsplit("/", maxsplit=1)[-1]
    parts = filename_with_context.split("__", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Invalid pipeline image path. Expected "
            "'images/{split}__{instance}__{filename}', got: "
            f"{pipeline_image!r}"
        )

    split, instance, filename = parts
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported InspecSafe split in image path: {split!r}")

    level_match = re.search(r"Level0?([1-4])(?:-|$)", instance, re.IGNORECASE)
    if level_match is None:
        raise ValueError(f"Cannot determine safety level from instance: {instance!r}")

    data_type = "Normal_data" if level_match.group(1) == "4" else "Anomaly_data"
    return (
        Path(data_root)
        / split
        / "Annotations"
        / data_type
        / instance
        / filename
    )
