#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert local InspecSafe-V1 dataset to CSV:

id,image_path,caption,safe_label
0001,/path/to/image.jpg,"...",safe
0002,/path/to/image.jpg,"...",unsafe

Usage:
    python build_inspecsafe_csv.py \
        --data_root /path/to/DATA_PATH \
        --output_csv /path/to/output.csv \
        --split all

Options:
    --split train / test / all
"""

import argparse
import csv
import json
import re
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def clean_caption(txt: str) -> str:
    """
    Try to keep only the semantic scene description.
    Some txt files may contain both description and final safety level.
    This function removes obvious safety-level headings/lines.
    """
    if not txt:
        return ""

    text = txt.strip()

    # If the file uses sections like [Image Description] / [Safety Level],
    # keep the part before [Safety Level].
    text = re.split(r"\[?\s*Safety\s*Level\s*\]?", text, flags=re.IGNORECASE)[0]

    # Remove section title if present.
    text = re.sub(r"\[?\s*Image\s*Description\s*\]?", "", text, flags=re.IGNORECASE)

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Remove the final line if it only looks like a class label.
    label_patterns = {
        "level one",
        "level two",
        "level three",
        "no abnormalities observed",
        "unrecognizable",
        "normal",
        "anomaly",
        "safe",
        "unsafe",
    }

    if lines:
        last = lines[-1].strip().lower().rstrip(".")
        if last in label_patterns:
            lines = lines[:-1]

    return " ".join(lines).strip()


def safe_label_from_folder(data_type: str) -> str:
    """
    Convert InspecSafe folder type to binary safe label.
    Normal_data  -> safe
    Anomaly_data -> unsafe
    """
    if data_type == "Normal_data":
        return "safe"
    if data_type == "Anomaly_data":
        return "unsafe"
    raise ValueError(f"Unknown data type folder: {data_type}")


def get_json_safety_label(json_path: Path) -> str:
    """
    Optional helper: not used for binary safe/unsafe by default.
    It tries to find a safety label in JSON if you later want multi-class labels.
    """
    if not json_path.exists():
        return ""

    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""

    possible_keys = [
        "safety_label",
        "safe_label",
        "safety_level",
        "level",
        "label",
        "status",
    ]

    def search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in possible_keys and isinstance(v, (str, int, float)):
                    return str(v)
                found = search(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = search(item)
                if found:
                    return found
        return ""

    return search(data)


def collect_samples(data_root: Path, split: str):
    """
    Scan:
        DATA_PATH/{train,test}/Annotations/{Normal_data,Anomaly_data}/.../*.jpg
    """
    splits = ["train", "test"] if split == "all" else [split]
    samples = []

    for sp in splits:
        annotations_root = data_root / sp / "Annotations"

        for data_type in ["Normal_data", "Anomaly_data"]:
            class_root = annotations_root / data_type
            if not class_root.exists():
                print(f"[WARN] Missing folder: {class_root}")
                continue

            for img_path in sorted(class_root.rglob("*")):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                txt_path = img_path.with_suffix(".txt")
                json_path = img_path.with_suffix(".json")

                raw_txt = read_text(txt_path)
                caption = clean_caption(raw_txt)

                samples.append({
                    "image_path": str(img_path),
                    "caption": caption,
                    "safe_label": safe_label_from_folder(data_type),
                    "split": sp,
                    "txt_path": str(txt_path) if txt_path.exists() else "",
                    "json_path": str(json_path) if json_path.exists() else "",
                })

    return samples


def write_csv(samples, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image_path", "caption", "safe_label"]
        )
        writer.writeheader()

        for idx, sample in enumerate(samples, start=1):
            writer.writerow({
                "id": f"{idx:04d}",
                "image_path": sample["image_path"],
                "caption": sample["caption"],
                "safe_label": sample["safe_label"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        help="Local InspecSafe-V1 DATA_PATH directory, containing train/ and test/"
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "test", "all"],
        help="Which split to export"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_csv = Path(args.output_csv)

    samples = collect_samples(data_root, args.split)

    if not samples:
        raise RuntimeError(
            f"No image samples found under {data_root}. "
            f"Please check DATA_PATH/{args.split}/Annotations structure."
        )

    write_csv(samples, output_csv)

    safe_count = sum(1 for s in samples if s["safe_label"] == "safe")
    unsafe_count = sum(1 for s in samples if s["safe_label"] == "unsafe")

    print(f"[OK] CSV saved to: {output_csv}")
    print(f"[OK] Total samples: {len(samples)}")
    print(f"[OK] safe: {safe_count}")
    print(f"[OK] unsafe: {unsafe_count}")


if __name__ == "__main__":
    main()

"""
python build_inspecsafe_csv.py \
  --data_root /root/autodl-tmp/data/inspecsafe \
  --output_csv /data/InspecSafe/dataset.csv \
  --split test
"""
