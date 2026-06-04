"""Build the caption and image embedding indexes from the dataset."""

from pathlib import Path

import chromadb
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

from config import (
    CAPTION_COLLECTION,
    CHROMA_DIR,
    CLIP_MODEL,
    DATASET_CSV,
    IMAGE_COLLECTION,
    PROJECT_ROOT,
    TEXT_EMBED_MODEL,
)


REQUIRED_COLUMNS = {"id", "image_path", "caption", "safe_label"}


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dataset() -> pd.DataFrame:
    if not DATASET_CSV.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")

    dataframe = pd.read_csv(DATASET_CSV, dtype={"id": str})
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


def build_indexes() -> None:
    dataframe = load_dataset()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    caption_collection = client.get_or_create_collection(
        CAPTION_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    image_collection = client.get_or_create_collection(
        IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    text_model = SentenceTransformer(TEXT_EMBED_MODEL)
    clip_model = SentenceTransformer(CLIP_MODEL)

    for row in dataframe.to_dict(orient="records"):
        item_id = str(row["id"]).strip()
        caption = str(row["caption"]).strip()
        safe_label = str(row["safe_label"]).strip()
        stored_image_path = str(row["image_path"]).strip()
        image_path = resolve_image_path(stored_image_path)

        if not item_id or not caption or not stored_image_path:
            raise ValueError(f"ID, image_path, and caption are required: {row}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for ID {item_id}: {image_path}")

        metadata = {
            "image_path": stored_image_path,
            "caption": caption,
            "safe_label": safe_label,
        }

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
                image_embedding = clip_model.encode(image).tolist()
        except UnidentifiedImageError as exc:
            raise ValueError(f"Invalid image for ID {item_id}: {image_path}") from exc

        caption_embedding = text_model.encode(caption).tolist()
        caption_collection.upsert(
            ids=[item_id],
            embeddings=[caption_embedding],
            documents=[caption],
            metadatas=[metadata],
        )
        image_collection.upsert(
            ids=[item_id],
            embeddings=[image_embedding],
            documents=[caption],
            metadatas=[metadata],
        )

    print(f"Built both indexes with {len(dataframe)} items in {CHROMA_DIR}.")


if __name__ == "__main__":
    build_indexes()

