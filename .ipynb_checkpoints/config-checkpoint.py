"""Central configuration for the Image RAG application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEMO_DIR = PROJECT_ROOT / "demo"

CAPTION_COLLECTION = "siglip2_caption_rag"
IMAGE_COLLECTION = "siglip2_image_rag"

# Complete local Hugging Face SigLIP2 snapshot.
EMBED_MODEL_PATH = "/root/autodl-tmp/model/siglip2"
EMBED_BATCH_SIZE = 128
EMBED_DEVICE = "auto"
RESET_COLLECTIONS_ON_BUILD = True

TOP_K = 5
MAX_TOP_K = 50

SAFETY_JUDGEMENT_TASK = "safety judgement"
CONSTRUCTIONSITE10K_TASK = "constructionsite10k"
LAB_SAFETY_TASK = "lab_safety"
LAB_SAFETY_GEN_TASK = "lab_safety_gen"

INSPECSAFE_DATASET = "inspecsafe"
CONSTRUCTIONSITE10K_DATASET = "constructionsite10k"
LAB_SAFETY_DATASET = "lab_safety"
LAB_SAFETY_GEN_DATASET = "lab_safety_gen"

SUPPORTED_RAG_DATASETS = {
    INSPECSAFE_DATASET,
    CONSTRUCTIONSITE10K_DATASET,
    LAB_SAFETY_DATASET,
    LAB_SAFETY_GEN_DATASET,
}

TASK_TO_RAG_DATASET = {
    SAFETY_JUDGEMENT_TASK: INSPECSAFE_DATASET,
    CONSTRUCTIONSITE10K_TASK: CONSTRUCTIONSITE10K_DATASET,
    LAB_SAFETY_TASK: LAB_SAFETY_DATASET,
    LAB_SAFETY_GEN_TASK: LAB_SAFETY_GEN_DATASET,
}

SUPPORTED_TASK_TYPES = {
    SAFETY_JUDGEMENT_TASK,
    CONSTRUCTIONSITE10K_TASK,
    LAB_SAFETY_TASK,
    LAB_SAFETY_GEN_TASK,
}
DEFAULT_SAFETY_QUERY = "Is the following image a safe scenario?"
DEFAULT_CONSTRUCTIONSITE10K_QUERY = "Inspect this construction site image."
DEFAULT_LAB_SAFETY_QUERY = "Answer the lab safety multiple-choice question."
DEFAULT_LAB_SAFETY_GEN_QUERY = "Classify this laboratory scene as hazardous or non-hazardous."

VLM_MODEL_PATH = "/root/autodl-tmp/model/qwenvl_2_5_3B"
# VLM_MODEL_PATH = "/root/autodl-tmp/model/gemma3_4b"
# VLM_MODEL_PATH = "/root/autodl-tmp/model/internvl2_4b"

VLM_PROCESSOR_PATH = VLM_MODEL_PATH
VLM_MAX_NEW_TOKENS = 2048
VLM_USE_FLASH_ATTENTION = False

# Model outputs are POSTed as UTF-8 text/plain after inference. Leave empty to
# disable forwarding. Example: "http://192.168.1.20:9000/response"
RESPONSE_FORWARD_URL = ""
RESPONSE_FORWARD_TIMEOUT_SECONDS = 5.0

SBERT_MODEL_PATH = "/root/autodl-tmp/model/all-MiniLM-L6-v2/sentence-transformers/all-MiniLM-L6-v2"
