"""Retrieval services for caption, image, and hybrid search."""

from functools import lru_cache
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CAPTION_COLLECTION,
    CHROMA_DIR,
    CLIP_MODEL,
    IMAGE_COLLECTION,
    MAX_TOP_K,
    TEXT_EMBED_MODEL,
    TOP_K,
)


SearchResult = dict[str, Any]


def _validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("Query must not be empty.")
    return normalized


def _validate_top_k(top_k: int) -> int:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    return top_k


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def _text_model() -> SentenceTransformer:
    return SentenceTransformer(TEXT_EMBED_MODEL)


@lru_cache(maxsize=1)
def _clip_model() -> SentenceTransformer:
    return SentenceTransformer(CLIP_MODEL)


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


def search_by_caption(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)
    collection = _collection(CAPTION_COLLECTION)
    item_count = collection.count()
    if item_count == 0:
        raise RuntimeError(f"Index collection '{CAPTION_COLLECTION}' is empty.")
    embedding = _text_model().encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, item_count),
    )
    return _format_results(results)


def search_by_image_embedding(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)
    collection = _collection(IMAGE_COLLECTION)
    item_count = collection.count()
    if item_count == 0:
        raise RuntimeError(f"Index collection '{IMAGE_COLLECTION}' is empty.")
    embedding = _clip_model().encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, item_count),
    )
    return _format_results(results)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[SearchResult]:
    """Fuse both ranked lists using reciprocal rank fusion."""
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)
    caption_results = search_by_caption(query, top_k)
    image_results = search_by_image_embedding(query, top_k)

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
