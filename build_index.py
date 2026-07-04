"""Build the caption and image embedding indexes from the dataset."""

import argparse
import json
import re
from pathlib import Path
from typing import Any
import time
from tqdm import tqdm

import chromadb
import pandas as pd
from PIL import Image, UnidentifiedImageError

from config import (
    CAPTION_COLLECTION,
    CHROMA_DIR,
    CONSTRUCTIONSITE10K_DATASET,
    EMBED_BATCH_SIZE,
    EMBED_MODEL_PATH,
    IMAGE_COLLECTION,
    INSPECSAFE_DATA_ROOT,
    INSPECSAFE_DATASET,
    LAB_SAFETY_DATASET,
    LAB_SAFETY_GEN_DATASET,
    PROJECT_ROOT,
    RESET_COLLECTIONS_ON_BUILD,
)
from embedding import encode_documents, encode_images
from utils.inspecsafe_paths import pipeline_image_to_dataset_path


REQUIRED_COLUMNS = {"id", "image_path", "caption", "safe_label"}
INSPECSAFE_INPUT_FORMAT_CSV = "csv"
INSPECSAFE_INPUT_FORMAT_PIPELINE = "inspecsafe_pipeline"


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset(dataset_csv: Path) -> pd.DataFrame:
    if not dataset_csv.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv}")

    dataframe = pd.read_csv(dataset_csv, dtype={"id": str})
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataframe = dataframe.fillna("")
    if dataframe.empty:
        raise ValueError("Dataset is empty.")
    if dataframe["id"].duplicated().any():
        raise ValueError("Dataset IDs must be unique.")

    return dataframe


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _message_content(sample: dict[str, Any], role: str) -> str:
    for message in sample.get("messages", []):
        if message.get("role") == role:
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    str(item.get("text", "")).strip()
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and str(item.get("text", "")).strip()
                )
    return ""


def load_inspecsafe_pipeline_dataset(
    dataset_json: Path,
    data_root: str | Path = INSPECSAFE_DATA_ROOT,
) -> pd.DataFrame:
    """Load pipeline_train.json into the common InspecSafe index schema."""
    if not dataset_json.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_json}")

    with dataset_json.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list) or not samples:
        raise ValueError("InspecSafe pipeline JSON must contain a non-empty list.")

    rows: list[dict[str, str]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Pipeline sample {index} must be a JSON object.")

        raw_image = str(sample.get("image", "")).strip().replace("\\", "/")
        image_path = pipeline_image_to_dataset_path(raw_image, data_root)
        assistant_content = _message_content(sample, "assistant")
        ground_truth = _parse_json_object(assistant_content)
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        scene_description = str(
            ground_truth.get("scene_description")
            or metadata.get("scene_description")
            or ""
        ).strip()
        hazards = ground_truth.get("hazards", metadata.get("hazards", []))
        if not isinstance(hazards, list):
            hazards = [str(hazards)] if str(hazards).strip() else []
        hazards = [str(hazard).strip() for hazard in hazards if str(hazard).strip()]

        safety_level = str(metadata.get("safety_level", "")).strip()
        overall_safety_level = str(
            ground_truth.get("overall_safety_level")
            or metadata.get("overall_safety_level")
            or ""
        ).strip()
        level_aliases = {
            "level i": "Level I",
            "level 1": "Level I",
            "level one": "Level I",
            "level ii": "Level II",
            "level 2": "Level II",
            "level two": "Level II",
            "level iii": "Level III",
            "level 3": "Level III",
            "level three": "Level III",
            "level iv": "Level IV",
            "level 4": "Level IV",
            "level four": "Level IV",
        }
        normalized_level = level_aliases.get(overall_safety_level.lower())
        if normalized_level is None:
            safety_level_match = re.fullmatch(
                r"level0?([1-4])",
                safety_level,
                re.IGNORECASE,
            )
            if safety_level_match:
                normalized_level = {
                    "1": "Level I",
                    "2": "Level II",
                    "3": "Level III",
                    "4": "Level IV",
                }[safety_level_match.group(1)]
        if normalized_level is None:
            raise ValueError(
                f"Pipeline sample {index} has no valid overall safety level."
            )
        overall_safety_level = normalized_level
        is_normal = overall_safety_level == "Level IV"
        split = str(metadata.get("split", "")).strip()
        if not split:
            split = raw_image.rsplit("/", maxsplit=1)[-1].split("__", maxsplit=1)[0]

        caption = scene_description or assistant_content
        if not caption:
            caption = f"InspecSafe scene classified as {overall_safety_level}."

        rows.append(
            {
                "id": Path(raw_image.rsplit("/", maxsplit=1)[-1]).stem,
                "image_path": str(image_path),
                "caption": caption,
                "safe_label": "safe" if is_normal else "unsafe",
                "safety_level": safety_level,
                "overall_safety_level": overall_safety_level,
                "scene_description": scene_description,
                "hazards": json.dumps(hazards, ensure_ascii=False),
                "split": split,
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe["id"].duplicated().any():
        raise ValueError("InspecSafe pipeline image IDs must be unique.")
    return dataframe


def load_inspecsafe_dataset(
    dataset_path: Path,
    input_format: str = INSPECSAFE_INPUT_FORMAT_CSV,
    data_root: str | Path = INSPECSAFE_DATA_ROOT,
) -> pd.DataFrame:
    """Load either legacy CSV or pipeline JSON InspecSafe input."""
    normalized_format = input_format.strip().lower().replace("-", "_")
    if normalized_format == INSPECSAFE_INPUT_FORMAT_CSV:
        return load_dataset(dataset_path)
    if normalized_format == INSPECSAFE_INPUT_FORMAT_PIPELINE:
        return load_inspecsafe_pipeline_dataset(dataset_path, data_root)
    raise ValueError(
        "Unsupported InspecSafe input format. Expected one of: "
        f"{INSPECSAFE_INPUT_FORMAT_CSV}, {INSPECSAFE_INPUT_FORMAT_PIPELINE}."
    )


def load_constructionsite10k_dataset(dataset_json: Path) -> pd.DataFrame:
    """Load ConstructionSite-10K JSON into the index CSV schema."""
    if not dataset_json.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_json}")

    with dataset_json.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list) or not samples:
        raise ValueError("ConstructionSite-10K JSON must contain a non-empty list.")

    rows: list[dict[str, str]] = []
    image_root = dataset_json.parent
    for sample in samples:
        raw_image = str(sample.get("image", "")).replace("\\", "/")
        image_path = image_root / raw_image
        ground_truth = _parse_json_object(_message_content(sample, "assistant"))
        violations = ground_truth.get("violations", [])
        if not isinstance(violations, list):
            violations = []
        rules = sorted({
            int(item["rule"])
            for item in violations
            if isinstance(item, dict) and str(item.get("rule", "")).isdigit()
        })
        annotation = str(ground_truth.get("annotation", "")).strip()
        rows.append(
            {
                "id": Path(raw_image).stem,
                "image_path": str(image_path),
                "caption": annotation or _message_content(sample, "assistant"),
                "safe_label": "unsafe" if rules else "safe",
                "violation_rules": ",".join(str(rule) for rule in rules) or "none",
                "violations_json": json.dumps(
                    violations,
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )

    return pd.DataFrame(rows)


def load_labsafety_dataset(dataset_json: Path) -> pd.DataFrame:
    """Load Lab Safety JSON into the index CSV schema."""
    if not dataset_json.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_json}")

    with dataset_json.open("r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Lab Safety JSON must contain a non-empty list.")

    rows: list[dict[str, str]] = []
    image_root = dataset_json.parent
    for index, sample in enumerate(samples):
        raw_image = str(sample.get("image", "")).replace("\\", "/")
        image_path = image_root / raw_image
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        question = _message_content(sample, "user")
        assistant_answer = _message_content(sample, "assistant")
        answer = str(metadata.get("answer") or assistant_answer).strip().upper()
        explanation = str(metadata.get("explanation", "")).strip()
        category = metadata.get("category", [])
        if isinstance(category, list):
            category_text = ", ".join(str(item) for item in category)
        else:
            category_text = str(category)
        level = str(metadata.get("level", "")).strip()

        caption_parts = [
            f"Question: {question}",
            f"Correct answer: {answer}",
        ]
        if explanation:
            caption_parts.append(f"Explanation: {explanation}")
        if category_text:
            caption_parts.append(f"Category: {category_text}")
        if level:
            caption_parts.append(f"Level: {level}")

        rows.append(
            {
                "id": f"{Path(raw_image).stem}_{index:06d}",
                "image_path": str(image_path),
                "caption": "\n".join(caption_parts),
                "safe_label": answer,
                "question": question,
                "answer": answer,
                "explanation": explanation,
                "category": category_text,
                "level": level,
            }
        )

    return pd.DataFrame(rows)


def load_labsafety_gen_dataset(dataset_jsonl: Path, split: str = "train") -> pd.DataFrame:
    """Load LabSafety-v1 JSONL into the index CSV schema."""
    if not dataset_jsonl.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_jsonl}")

    split = split.strip().lower()
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all.")

    rows: list[dict[str, str]] = []
    with dataset_jsonl.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            sample_split = str(sample.get("split", "")).strip().lower()
            if split != "all" and sample_split != split:
                continue

            raw_image = str(sample.get("image", "")).replace("\\", "/")
            image_path = dataset_jsonl.parent / raw_image
            image_id = str(sample.get("image_id") or Path(raw_image).stem).strip()
            safety_label = str(sample.get("safety_label", "")).strip().lower()
            description = str(sample.get("description", "")).strip()
            hazards = sample.get("hazards", [])
            if isinstance(hazards, list):
                hazards_text = "; ".join(str(item) for item in hazards)
            else:
                hazards_text = str(hazards).strip()
            vlm_label = str(sample.get("vlm_label", "")).strip().lower()
            agree = str(sample.get("agree", "")).strip()

            caption_parts = [
                f"Description: {description}",
                f"Safety label: {safety_label}",
            ]
            if hazards_text:
                caption_parts.append(f"Hazards: {hazards_text}")
            if vlm_label:
                caption_parts.append(f"VLM label: {vlm_label}")
            if agree:
                caption_parts.append(f"Agreement flag: {agree}")

            rows.append(
                {
                    "id": image_id or f"line_{line_number:06d}",
                    "image_path": str(image_path),
                    "caption": "\n".join(caption_parts),
                    "safe_label": safety_label,
                    "description": description,
                    "hazards": hazards_text,
                    "vlm_label": vlm_label,
                    "agree": agree,
                    "split": sample_split,
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError(f"No LabSafety-v1 rows found for split: {split}.")
    return dataframe


def build_indexes_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset: str,
) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataframe = dataframe.fillna("")
    if dataframe.empty:
        raise ValueError("Dataset is empty.")
    if dataframe["id"].duplicated().any():
        raise ValueError("Dataset IDs must be unique.")

    index_dir = CHROMA_DIR / dataset
    client = chromadb.PersistentClient(path=str(index_dir))

    if RESET_COLLECTIONS_ON_BUILD:
        existing_collections = {
            item if isinstance(item, str) else item.name
            for item in client.list_collections()
        }
        for collection_name in (CAPTION_COLLECTION, IMAGE_COLLECTION):
            if collection_name in existing_collections:
                client.delete_collection(collection_name)

    collection_metadata = {
        "hnsw:space": "cosine",
        "embedding_model": EMBED_MODEL_PATH,
    }
    caption_collection = client.get_or_create_collection(
        CAPTION_COLLECTION,
        metadata=collection_metadata,
    )
    image_collection = client.get_or_create_collection(
        IMAGE_COLLECTION,
        metadata=collection_metadata,
    )

    rows = dataframe.to_dict(orient="records")
    total_batches = (len(rows) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    start_time = time.time()
    
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for start in range(0, len(rows), EMBED_BATCH_SIZE):
            batch = rows[start : start + EMBED_BATCH_SIZE]
            ids: list[str] = []
            captions: list[str] = []
            images: list[Image.Image] = []
            metadatas: list[dict[str, str]] = []

            for row in batch:
                item_id = str(row["id"]).strip()
                caption = str(row["caption"]).strip()
                safe_label = str(row["safe_label"]).strip()
                stored_image_path = str(row["image_path"]).strip()
                image_path = resolve_image_path(stored_image_path)

                if not item_id or not caption or not stored_image_path:
                    raise ValueError(f"ID, image_path, and caption are required: {row}")
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found for ID {item_id}: {image_path}")

                try:
                    with Image.open(image_path) as source_image:
                        images.append(source_image.convert("RGB"))
                except UnidentifiedImageError as exc:
                    raise ValueError(f"Invalid image for ID {item_id}: {image_path}") from exc

                ids.append(item_id)
                captions.append(caption)
                metadatas.append(
                    {
                        "image_path": stored_image_path,
                        "caption": caption,
                        "safe_label": safe_label,
                        **{
                            key: str(row[key])
                            for key in (
                                "violation_rules",
                                "violations_json",
                                "question",
                                "answer",
                                "explanation",
                                "category",
                                "level",
                                "safety_level",
                                "overall_safety_level",
                                "scene_description",
                                "description",
                                "hazards",
                                "vlm_label",
                                "agree",
                                "split",
                            )
                            if key in row and str(row[key]).strip()
                        },
                    }
                )

            caption_embeddings = encode_documents(captions)
            image_embeddings = encode_images(images)
            caption_collection.upsert(
                ids=ids,
                embeddings=caption_embeddings,
                documents=captions,
                metadatas=metadatas,
            )
            image_collection.upsert(
                ids=ids,
                embeddings=image_embeddings,
                documents=captions,
                metadatas=metadatas,
            )
            elapsed = time.time() - start_time
            pbar.set_description(f"Batch start={start} | elapsed={elapsed:.2f}s")
            pbar.update(1)

    print(f"Built both indexes with {len(dataframe)} items in {index_dir}.")


def build_indexes(
    dataset_path: Path,
    input_format: str = INSPECSAFE_INPUT_FORMAT_CSV,
    data_root: str | Path = INSPECSAFE_DATA_ROOT,
) -> None:
    build_indexes_from_dataframe(
        load_inspecsafe_dataset(dataset_path, input_format, data_root),
        dataset=INSPECSAFE_DATASET,
    )


def build_constructionsite10k_indexes(dataset_json: Path) -> None:
    build_indexes_from_dataframe(
        load_constructionsite10k_dataset(dataset_json),
        dataset=CONSTRUCTIONSITE10K_DATASET,
    )


def build_labsafety_indexes(dataset_json: Path) -> None:
    build_indexes_from_dataframe(
        load_labsafety_dataset(dataset_json),
        dataset=LAB_SAFETY_DATASET,
    )


def build_labsafety_gen_indexes(dataset_jsonl: Path, split: str) -> None:
    build_indexes_from_dataframe(
        load_labsafety_gen_dataset(dataset_jsonl, split=split),
        dataset=LAB_SAFETY_GEN_DATASET,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build caption and image indexes from a supported dataset file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-input",
        "--dataset-csv",
        "--dataset_csv",
        dest="dataset_input",
        type=Path,
        help="Path to an InspecSafe CSV or pipeline JSON input file.",
    )
    source.add_argument(
        "--constructionsite-json",
        type=Path,
        help="Path to ConstructionSite-10K train.json.",
    )
    source.add_argument(
        "--lab-safety-json",
        type=Path,
        help="Path to Lab Safety train JSON.",
    )
    source.add_argument(
        "--lab-safety-gen-jsonl",
        type=Path,
        help="Path to LabSafety-v1 annotations.jsonl.",
    )
    parser.add_argument(
        "--input-format",
        "--input-data-format",
        "--input_data_format",
        choices=[INSPECSAFE_INPUT_FORMAT_CSV, INSPECSAFE_INPUT_FORMAT_PIPELINE],
        default=INSPECSAFE_INPUT_FORMAT_CSV,
        help="Format used by --dataset-input (default: csv).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(INSPECSAFE_DATA_ROOT),
        help="Original InspecSafe DATA_PATH root used for pipeline JSON images.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="train",
        help="Split to index for JSONL datasets that include a split field.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.constructionsite_json:
        build_constructionsite10k_indexes(args.constructionsite_json)
    elif args.lab_safety_json:
        build_labsafety_indexes(args.lab_safety_json)
    elif args.lab_safety_gen_jsonl:
        build_labsafety_gen_indexes(args.lab_safety_gen_jsonl, args.split)
    else:
        build_indexes(args.dataset_input, args.input_format, args.data_root)
