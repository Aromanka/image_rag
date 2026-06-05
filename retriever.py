"""Retrieval services for caption, image, and hybrid search."""

from functools import lru_cache
from pathlib import Path
import re
import shutil
from typing import Any

import chromadb
from PIL import Image, UnidentifiedImageError

from config import (
    CAPTION_COLLECTION,
    CHROMA_DIR,
    DEMO_DIR,
    IMAGE_COLLECTION,
    MAX_TOP_K,
    PROJECT_ROOT,
    TOP_K,
)
from embedding import encode_images, encode_query


SearchResult = dict[str, Any]


def save_retrieved_images(results: list[SearchResult]) -> list[SearchResult]:
    """Copy retrieved images directly into the flat demo directory."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    saved_results: list[SearchResult] = []

    for rank, result in enumerate(results, start=1):
        stored_path = str(result.get("image_path", "")).strip()
        source_path = Path(stored_path)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Retrieved image not found for ID {result.get('id', '')}: "
                f"{source_path}"
            )

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.get("id", "")))
        destination = DEMO_DIR / f"{rank:02d}_{safe_id}_{source_path.name}"
        shutil.copy2(source_path, destination)
        saved_results.append(
            {
                **result,
                "demo_path": str(destination.relative_to(PROJECT_ROOT)),
            }
        )

    return saved_results


def _validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("Query must not be empty.")
    return normalized


def _validate_top_k(top_k: int) -> int:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    return top_k


def resolve_query_image_path(query_image: str | Path) -> Path:
    path = Path(query_image).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Query image not found: {path}")
    return path


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(name: str):
    try:
        return _client().get_collection(name)
    except Exception as exc:
        raise RuntimeError(
            f"Index collection '{name}' is unavailable. Build the indexes first."
        ) from exc


def _format_results(results: dict[str, Any]) -> list[SearchResult]:
    output: list[SearchResult] = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for item_id, distance, metadata in zip(ids, distances, metadatas):
        output.append(
            {
                "id": item_id,
                "distance": float(distance),
                "image_path": metadata.get("image_path", ""),
                "caption": metadata.get("caption", ""),
                "safe_label": metadata.get("safe_label", ""),
            }
        )
    return output


def _search_collection(
    collection_name: str,
    embedding: list[float],
    top_k: int,
) -> list[SearchResult]:
    collection = _collection(collection_name)
    item_count = collection.count()
    if item_count == 0:
        raise RuntimeError(f"Index collection '{collection_name}' is empty.")
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, item_count),
    )
    return _format_results(results)


def search_by_caption(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)
    return _search_collection(CAPTION_COLLECTION, encode_query(query), top_k)


def search_by_image_embedding(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)
    return _search_collection(IMAGE_COLLECTION, encode_query(query), top_k)


def search_by_query_image(
    query_image: str | Path,
    top_k: int = TOP_K,
) -> list[SearchResult]:
    """Retrieve visually similar examples for a local query image."""
    top_k = _validate_top_k(top_k)
    image_path = resolve_query_image_path(query_image)
    try:
        with Image.open(image_path) as source_image:
            embedding = encode_images([source_image.convert("RGB")])[0]
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid query image: {image_path}") from exc

    return _search_collection(IMAGE_COLLECTION, embedding, top_k)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Fuse both ranked lists using reciprocal rank fusion."""
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)
    embedding = encode_query(query)
    caption_results = _search_collection(CAPTION_COLLECTION, embedding, top_k)
    image_results = _search_collection(IMAGE_COLLECTION, embedding, top_k)

    fused: dict[str, SearchResult] = {}
    for source, results in (
        ("caption", caption_results),
        ("image", image_results),
    ):
        for rank, result in enumerate(results, start=1):
            item_id = result["id"]
            if item_id not in fused:
                fused[item_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "matched_by": [],
                }
            fused[item_id]["rrf_score"] += 1.0 / (60 + rank)
            fused[item_id]["matched_by"].append(source)

    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    return ranked[:top_k]
