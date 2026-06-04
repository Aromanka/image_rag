"""Build the caption and image embedding indexes from the dataset."""

import argparse
from pathlib import Path

import chromadb
import pandas as pd
from PIL import Image, UnidentifiedImageError

from config import (
    CAPTION_COLLECTION,
    CHROMA_DIR,
    EMBED_BATCH_SIZE,
    EMBED_MODEL_PATH,
    IMAGE_COLLECTION,
    PROJECT_ROOT,
    RESET_COLLECTIONS_ON_BUILD,
)
from embedding import encode_documents, encode_images


REQUIRED_COLUMNS = {"id", "image_path", "caption", "safe_label"}


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


def build_indexes(dataset_csv: Path) -> None:
    dataframe = load_dataset(dataset_csv)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

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

    print(f"Built both indexes with {len(dataframe)} items in {CHROMA_DIR}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build caption and image indexes from a dataset CSV."
    )
    parser.add_argument(
        "--dataset-csv",
        "--dataset_csv",
        required=True,
        type=Path,
        help="Path to the dataset CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_indexes(parse_args().dataset_csv)
