"""Central configuration for the Image RAG application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

CAPTION_COLLECTION = "caption_rag"
IMAGE_COLLECTION = "image_rag"

TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"

TOP_K = 5
MAX_TOP_K = 50

