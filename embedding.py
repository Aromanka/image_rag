"""Shared jina-clip-v2 encoder for text and images."""

from functools import lru_cache
from typing import Sequence

from PIL import Image
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_BATCH_SIZE,
    EMBED_MODEL_PATH,
    EMBED_NORMALIZE,
    EMBED_TRUNCATE_DIM,
    EMBED_TRUST_REMOTE_CODE,
    TEXT_QUERY_PROMPT,
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        EMBED_MODEL_PATH,
        trust_remote_code=EMBED_TRUST_REMOTE_CODE,
        truncate_dim=EMBED_TRUNCATE_DIM,
    )


def encode_documents(texts: Sequence[str]) -> list[list[float]]:
    """Encode caption texts stored as retrieval documents."""
    embeddings = get_embedding_model().encode(
        list(texts),
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=EMBED_NORMALIZE,
    )
    return embeddings.tolist()


def encode_query(text: str) -> list[float]:
    """Encode a text query using jina-clip-v2's retrieval query prompt."""
    embedding = get_embedding_model().encode(
        text,
        prompt_name=TEXT_QUERY_PROMPT,
        normalize_embeddings=EMBED_NORMALIZE,
    )
    return embedding.tolist()


def encode_images(images: Sequence[Image.Image]) -> list[list[float]]:
    """Encode images into the same multimodal vector space."""
    embeddings = get_embedding_model().encode(
        list(images),
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=EMBED_NORMALIZE,
    )
    return embeddings.tolist()
