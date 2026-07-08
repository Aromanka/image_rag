"""Central configuration for the Image RAG application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEMO_DIR = PROJECT_ROOT / "demo"

# Server-side root of the original InspecSafe-V1 directory tree.
INSPECSAFE_DATA_ROOT = "/root/autodl-tmp/data/inspecsafe/DATA_PATH"

CAPTION_COLLECTION = "siglip2_caption_rag"
IMAGE_COLLECTION = "siglip2_image_rag"

# Complete local Hugging Face SigLIP2 snapshot.
EMBED_MODEL_PATH = "/root/autodl-tmp/model/siglip2"
EMBED_BATCH_SIZE = 128
EMBED_DEVICE = "auto"
RESET_COLLECTIONS_ON_BUILD = True

TOP_K = 5
MAX_TOP_K = 50
GATED_RAG = 0.0

SAFETY_JUDGEMENT_TASK = "safety judgement"
SAFETY_LEVEL_TASK = "safety level"
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
    SAFETY_LEVEL_TASK: INSPECSAFE_DATASET,
    CONSTRUCTIONSITE10K_TASK: CONSTRUCTIONSITE10K_DATASET,
    LAB_SAFETY_TASK: LAB_SAFETY_DATASET,
    LAB_SAFETY_GEN_TASK: LAB_SAFETY_GEN_DATASET,
}

SUPPORTED_TASK_TYPES = {
    SAFETY_JUDGEMENT_TASK,
    SAFETY_LEVEL_TASK,
    CONSTRUCTIONSITE10K_TASK,
    LAB_SAFETY_TASK,
    LAB_SAFETY_GEN_TASK,
}
DEFAULT_SAFETY_QUERY = "Is the following image a safe scenario?"
DEFAULT_SAFETY_QUERY_2 = """
You are evaluating the overall safety condition of a construction site based on clearly visible evidence.

A scene is UNSAFE only when the image **clearly** and **definitely** shows unsafe actions, missing required protective equipment, or uncontrolled hazardous conditions.

Otherwise, if there is no strong evidence, it is safe!

Do not infer hidden risks.
Do not assume missing information.
Construction activity alone does not imply an unsafe scene.

Do not speculate about risks outside the image.
"""
DEFAULT_SAFETY_LEVEL_QUERY = "Assess the safety level of this inspection image."
DEFAULT_CONSTRUCTIONSITE10K_QUERY = "Inspect this construction site image."
DEFAULT_LAB_SAFETY_QUERY = "Answer the lab safety multiple-choice question."
DEFAULT_LAB_SAFETY_GEN_QUERY = "Classify this laboratory scene as hazardous or non-hazardous."

VLM_MODEL_PATH = "/root/autodl-tmp/model/qwenvl_2_5_3B"
# VLM_MODEL_PATH = "/root/autodl-tmp/model/gemma3_4b"
# VLM_MODEL_PATH = "/root/autodl-tmp/model/internvl2_4b"

VLM_PROCESSOR_PATH = VLM_MODEL_PATH
# VLM_LORA_WEIGHTS = "lora_weights/gemma3_4b_lora_v2"
# VLM_LORA_WEIGHTS = "lora_weights/gemma3_4b_lora_lab_v1"
VLM_LORA_WEIGHTS = ""
VLM_MAX_NEW_TOKENS = 2048
VLM_USE_FLASH_ATTENTION = False

# The structured InspecSafe safety-level response includes a scene description,
# hazard list, and four-level classification.
INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS = 384

# Two-stage InspecSafe inference limits. The first pass is deliberately short
# because it is only a gate; the second pass runs only after an unsafe result.
INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS = 8
INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS = 128

# Model outputs are POSTed as UTF-8 text/plain after inference. Leave empty to
# disable forwarding. Example: "http://192.168.1.20:9000/response"
RESPONSE_FORWARD_URL = ""
RESPONSE_FORWARD_TIMEOUT_SECONDS = 5.0

SBERT_MODEL_PATH = "/root/autodl-tmp/model/all-MiniLM-L6-v2/sentence-transformers/all-MiniLM-L6-v2"
