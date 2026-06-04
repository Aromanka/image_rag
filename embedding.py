"""Shared jina-clip-v2 encoder for text and images."""

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

# Never let Hugging Face or Transformers fetch missing model files or code.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# This inference-only application does not use Weights & Biases. Some timm
# versions import it opportunistically, and a broken optional wandb install
# must not prevent the local embedding model from loading.
sys.modules["wandb"] = None

import torch
from PIL import Image
from transformers import AutoModel

from config import (
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL_PATH,
    EMBED_NORMALIZE,
    EMBED_TRUNCATE_DIM,
    EMBED_TRUST_REMOTE_CODE,
    TEXT_QUERY_PROMPT,
)


def _is_jina_clip_config(config_path: Path) -> bool:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return (
        config.get("model_type") == "jina_clip"
        and "AutoModel" in config.get("auto_map", {})
    )


@lru_cache(maxsize=1)
def resolve_model_path() -> str:
    """Resolve a direct model directory or a nested Hugging Face snapshot."""
    configured_path = Path(EMBED_MODEL_PATH).expanduser()
    if not configured_path.is_dir():
        raise FileNotFoundError(
            f"Jina CLIP model directory does not exist: {configured_path}"
        )

    direct_config = configured_path / "config.json"
    if _is_jina_clip_config(direct_config):
        return str(configured_path)

    snapshot_configs = sorted(configured_path.glob("**/snapshots/*/config.json"))
    other_configs = sorted(configured_path.glob("**/config.json"))
    for config_path in snapshot_configs + other_configs:
        if _is_jina_clip_config(config_path):
            return str(config_path.parent)

    raise ValueError(
        "No valid jina-clip-v2 snapshot was found under "
        f"{configured_path}. Its model config must contain "
        "'model_type': 'jina_clip' and an AutoModel entry in 'auto_map'. "
        "Point EMBED_MODEL_PATH at the complete Hugging Face model snapshot."
    )


def _embedding_device() -> str:
    if EMBED_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return EMBED_DEVICE


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load the official Jina CLIP Transformers model."""
    model = AutoModel.from_pretrained(
        resolve_model_path(),
        trust_remote_code=EMBED_TRUST_REMOTE_CODE,
        local_files_only=True,
    )
    if not hasattr(model, "encode_text") or not hasattr(model, "encode_image"):
        raise TypeError(
            "Loaded model does not expose jina-clip-v2 encode_text/encode_image."
        )
    return model.eval()


def encode_documents(texts: Sequence[str]) -> list[list[float]]:
    """Encode caption texts stored as retrieval documents."""
    embeddings = get_embedding_model().encode_text(
        list(texts),
        batch_size=EMBED_BATCH_SIZE,
        device=_embedding_device(),
        normalize_embeddings=EMBED_NORMALIZE,
        truncate_dim=EMBED_TRUNCATE_DIM,
    )
    return embeddings.tolist()


def encode_query(text: str) -> list[float]:
    """Encode a text query using jina-clip-v2's retrieval query task."""
    embedding = get_embedding_model().encode_text(
        text,
        task=TEXT_QUERY_PROMPT,
        device=_embedding_device(),
        normalize_embeddings=EMBED_NORMALIZE,
        truncate_dim=EMBED_TRUNCATE_DIM,
    )
    return embedding.tolist()


def encode_images(images: Sequence[Image.Image]) -> list[list[float]]:
    """Encode images into the same multimodal vector space."""
    embeddings = get_embedding_model().encode_image(
        list(images),
        batch_size=EMBED_BATCH_SIZE,
        device=_embedding_device(),
        normalize_embeddings=EMBED_NORMALIZE,
        truncate_dim=EMBED_TRUNCATE_DIM,
    )
    return embeddings.tolist()
