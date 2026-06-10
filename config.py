"""Central configuration for the Image RAG application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEMO_DIR = PROJECT_ROOT / "demo"

CAPTION_COLLECTION = "siglip2_caption_rag"
IMAGE_COLLECTION = "siglip2_image_rag"

# Complete local Hugging Face SigLIP2 snapshot.
EMBED_MODEL_PATH = "/root/autodl-tmp/model/siglip2"
EMBED_BATCH_SIZE = 32
EMBED_DEVICE = "auto"
RESET_COLLECTIONS_ON_BUILD = True

TOP_K = 5
MAX_TOP_K = 50

SAFETY_JUDGEMENT_TASK = "safety judgement"
CONSTRUCTIONSITE10K_TASK = "constructionsite10k"

SUPPORTED_TASK_TYPES = {SAFETY_JUDGEMENT_TASK, CONSTRUCTIONSITE10K_TASK}
DEFAULT_SAFETY_QUERY = "Is the following image a safe scenario?"
DEFAULT_CONSTRUCTIONSITE10K_QUERY = "Inspect this construction site image."

VLM_MODEL_PATH = "/root/autodl-tmp/model/qwenvl_2_5_3B"
VLM_PROCESSOR_PATH = VLM_MODEL_PATH
VLM_MAX_NEW_TOKENS = 2048
