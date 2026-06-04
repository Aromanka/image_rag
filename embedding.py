"""Shared local SigLIP2 encoder for text and images."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

# Never let Hugging Face or Transformers fetch missing model files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import EMBED_BATCH_SIZE, EMBED_DEVICE, EMBED_MODEL_PATH


def _is_siglip2_config(config_path: Path) -> bool:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return config.get("model_type") == "siglip2"


@lru_cache(maxsize=1)
def resolve_model_path() -> str:
    """Resolve a direct local model directory or nested local snapshot."""
    configured_path = Path(EMBED_MODEL_PATH).expanduser()
    if not configured_path.is_dir():
        raise FileNotFoundError(f"SigLIP2 model directory does not exist: {configured_path}")
    return str(configured_path)


@lru_cache(maxsize=1)
def _embedding_device() -> torch.device:
    configured_device = EMBED_DEVICE
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(configured_device)


@lru_cache(maxsize=1)
def get_embedding_processor() -> Any:
    """Load the SigLIP2 processor strictly from local files."""
    return AutoProcessor.from_pretrained(
        resolve_model_path(),
        local_files_only=True,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load the SigLIP2 model strictly from local files."""
    model = AutoModel.from_pretrained(
        resolve_model_path(),
        local_files_only=True,
    )
    if not hasattr(model, "get_text_features") or not hasattr(
        model, "get_image_features"
    ):
        raise TypeError(
            "Loaded model does not expose SigLIP2 get_text_features/"
            "get_image_features."
        )
    return model.to(_embedding_device()).eval()


def _feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    raise TypeError("SigLIP2 feature method returned an unsupported output type.")


def _encode_texts(texts: Sequence[str]) -> list[list[float]]:
    model = get_embedding_model()
    processor = get_embedding_processor()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        inputs = processor(
            text=batch,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(_embedding_device())
        with torch.inference_mode():
            features = _feature_tensor(model.get_text_features(**inputs))
            features = F.normalize(features, p=2, dim=-1)
        embeddings.extend(features.float().cpu().tolist())

    return embeddings


def encode_documents(texts: Sequence[str]) -> list[list[float]]:
    """Encode caption texts stored as retrieval documents."""
    return _encode_texts(texts)


def encode_query(text: str) -> list[float]:
    """Encode one text query into the shared SigLIP2 embedding space."""
    return _encode_texts([text])[0]


def encode_images(images: Sequence[Image.Image]) -> list[list[float]]:
    """Encode images into the shared SigLIP2 embedding space."""
    model = get_embedding_model()
    processor = get_embedding_processor()
    embeddings: list[list[float]] = []

    for start in range(0, len(images), EMBED_BATCH_SIZE):
        batch = list(images[start : start + EMBED_BATCH_SIZE])
        inputs = processor(images=batch, return_tensors="pt").to(_embedding_device())
        with torch.inference_mode():
            features = _feature_tensor(model.get_image_features(**inputs))
            features = F.normalize(features, p=2, dim=-1)
        embeddings.extend(features.float().cpu().tolist())

    return embeddings
