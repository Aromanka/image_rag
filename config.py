"""Central configuration for the Image RAG application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

CAPTION_COLLECTION = "jina_clip_v2_caption_rag"
IMAGE_COLLECTION = "jina_clip_v2_image_rag"

# Local Hugging Face snapshot of jinaai/jina-clip-v2.
EMBED_MODEL_PATH = "/root/autodl-tmp/model/jina_clip"
EMBED_TRUST_REMOTE_CODE = True
EMBED_NORMALIZE = True
EMBED_TRUNCATE_DIM = 1024
EMBED_BATCH_SIZE = 32
EMBED_DEVICE = "auto"
TEXT_QUERY_PROMPT = "retrieval.query"
RESET_COLLECTIONS_ON_BUILD = True

TOP_K = 5
MAX_TOP_K = 50
