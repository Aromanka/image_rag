"""Central configuration for the Image RAG application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

CAPTION_COLLECTION = "siglip2_caption_rag"
IMAGE_COLLECTION = "siglip2_image_rag"

# Complete local Hugging Face SigLIP2 snapshot.
EMBED_MODEL_PATH = "/root/autodl-tmp/model/siglip2"
EMBED_BATCH_SIZE = 32
EMBED_DEVICE = "auto"
RESET_COLLECTIONS_ON_BUILD = True

TOP_K = 5
MAX_TOP_K = 50
