"""VLM inference entry points with optional image RAG context."""

import argparse
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import (
    CONSTRUCTIONSITE10K_TASK,
    DEFAULT_SAFETY_QUERY,
    DEFAULT_SAFETY_LEVEL_QUERY,
    DEFAULT_CONSTRUCTIONSITE10K_QUERY,
    DEFAULT_LAB_SAFETY_GEN_QUERY,
    DEFAULT_LAB_SAFETY_QUERY,
    LAB_SAFETY_GEN_TASK,
    LAB_SAFETY_TASK,
    GATED_RAG,
    INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    PROJECT_ROOT,
    SAFETY_JUDGEMENT_TASK,
    SAFETY_LEVEL_TASK,
    SUPPORTED_TASK_TYPES,
    TASK_TO_RAG_DATASET,
    TOP_K,
    VLM_MAX_NEW_TOKENS,
    VLM_LORA_WEIGHTS,
    VLM_MODEL_PATH,
    VLM_PROCESSOR_PATH,
    VLM_USE_FLASH_ATTENTION,
)


QWEN25VL_BACKEND = "qwen2_5_vl"
GEMMA3_BACKEND = "gemma3"
INTERNVL_BACKEND = "internvl"
INTERNVL_IMG_SIZE = 448
INTERNVL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
INTERNVL_IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
_ACTIVE_LORA_WEIGHTS: Path | None = None


def _normalize_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    if isinstance(path, str) and not path.strip():
        return None
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    return raw_path


def configure_lora_weights(lora_weights: str | Path | None) -> None:
    """Configure optional PEFT LoRA weights before the VLM is first loaded."""
    global _ACTIVE_LORA_WEIGHTS

    normalized = _normalize_optional_path(lora_weights)
    if normalized is not None and not normalized.exists():
        raise FileNotFoundError(f"LoRA weights path not found: {normalized}")

    if normalized == _ACTIVE_LORA_WEIGHTS:
        return

    if _vlm_components.cache_info().currsize:
        raise RuntimeError(
            "Cannot change LoRA weights after the VLM has been loaded in this "
            "process. Configure --lora-weights before preloading or inference."
        )
    _ACTIVE_LORA_WEIGHTS = normalized


def active_lora_weights() -> str | None:
    return str(_ACTIVE_LORA_WEIGHTS) if _ACTIVE_LORA_WEIGHTS is not None else None


def add_lora_cli_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora-weights",
        "--lora_weights",
        dest="lora_weights",
        type=Path,
        default=None,
        help=(
            "Optional PEFT LoRA adapter directory or file. Relative paths are "
            "resolved from the project root."
        ),
    )


def _processor_path_for_lora() -> str | Path:
    if _ACTIVE_LORA_WEIGHTS is None or not _ACTIVE_LORA_WEIGHTS.is_dir():
        return VLM_PROCESSOR_PATH
    processor_markers = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    if any((_ACTIVE_LORA_WEIGHTS / marker).exists() for marker in processor_markers):
        return _ACTIVE_LORA_WEIGHTS
    return VLM_PROCESSOR_PATH


def _apply_lora_weights(model: Any) -> Any:
    if _ACTIVE_LORA_WEIGHTS is None:
        print(f'Not using LORA!')
        return model

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "LoRA weights require the 'peft' package. Install project "
            "dependencies after updating requirements.txt."
        ) from exc

    model = PeftModel.from_pretrained(
        model,
        str(_ACTIVE_LORA_WEIGHTS),
        is_trainable=False,
    )
    print(f"Loaded LoRA weights: {_ACTIVE_LORA_WEIGHTS}", flush=True)
    return model


def _validate_task_type(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized in {"safety_level", "safety-level"}:
        normalized = SAFETY_LEVEL_TASK
    if normalized not in SUPPORTED_TASK_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(f"Unsupported task_type '{task_type}'. Supported: {supported}.")
    return normalized


def _default_query_for_task(task_type: str) -> str:
    task_type = _validate_task_type(task_type)
    if task_type == CONSTRUCTIONSITE10K_TASK:
        return DEFAULT_CONSTRUCTIONSITE10K_QUERY
    if task_type == LAB_SAFETY_TASK:
        return DEFAULT_LAB_SAFETY_QUERY
    if task_type == LAB_SAFETY_GEN_TASK:
        return DEFAULT_LAB_SAFETY_GEN_QUERY
    if task_type == SAFETY_LEVEL_TASK:
        return DEFAULT_SAFETY_LEVEL_QUERY
    return DEFAULT_SAFETY_QUERY


def _resolve_query_image_path(query_image: str | Path) -> Path:
    path = Path(query_image).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Query image not found: {path}")
    return path


def _resolve_image_path(image_path: str | Path) -> Path:
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def _infer_vlm_backend() -> str:
    model_hint = f"{VLM_MODEL_PATH} {VLM_PROCESSOR_PATH}".lower()
    if "gemma" in model_hint:
        return GEMMA3_BACKEND
    if "internvl" in model_hint:
        return INTERNVL_BACKEND
    return QWEN25VL_BACKEND


def _build_internvl_transform() -> Any:
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(
            (INTERNVL_IMG_SIZE, INTERNVL_IMG_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=INTERNVL_IMAGENET_MEAN,
            std=INTERNVL_IMAGENET_STD,
        ),
    ])


def _prepare_internvl_optional_imports() -> None:
    """Avoid broken optional logging imports pulled in by InternVL/timm."""
    os.environ.setdefault("WANDB_DISABLED", "true")
    existing_wandb = sys.modules.get("wandb")
    if existing_wandb is not None and all(
        hasattr(existing_wandb, attr) for attr in ("init", "log", "finish")
    ):
        return

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.run = None
    wandb_stub.init = lambda *args, **kwargs: None
    wandb_stub.log = lambda *args, **kwargs: None
    wandb_stub.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub


@lru_cache(maxsize=1)
def _vlm_components() -> tuple[Any, Any, str, Any, Any]:
    import torch

    backend = _infer_vlm_backend()

    if backend == GEMMA3_BACKEND:
        from transformers import AutoProcessor
        from transformers import Gemma3ForConditionalGeneration

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if VLM_USE_FLASH_ATTENTION:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = Gemma3ForConditionalGeneration.from_pretrained(
            VLM_MODEL_PATH,
            **model_kwargs,
        )
        model = _apply_lora_weights(model)
        processor = AutoProcessor.from_pretrained(
            str(_processor_path_for_lora()),
            trust_remote_code=True,
        )
        model.eval()
        return model, processor, backend, None, torch

    if backend == INTERNVL_BACKEND:
        _prepare_internvl_optional_imports()

        from transformers import AutoModel, AutoTokenizer

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if VLM_USE_FLASH_ATTENTION:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        tokenizer = AutoTokenizer.from_pretrained(
            str(_processor_path_for_lora()),
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            VLM_MODEL_PATH,
            **model_kwargs,
        )
        model = _apply_lora_weights(model)
        img_context_token_id = tokenizer.convert_tokens_to_ids(
            INTERNVL_IMG_CONTEXT_TOKEN
        )
        if img_context_token_id is not None:
            model.img_context_token_id = img_context_token_id
        model.eval()
        return model, tokenizer, backend, _build_internvl_transform(), torch

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    if VLM_USE_FLASH_ATTENTION:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_PATH,
        **model_kwargs,
    )
    model = _apply_lora_weights(model)
    processor = AutoProcessor.from_pretrained(str(_processor_path_for_lora()))
    model.eval()
    return model, processor, backend, process_vision_info, torch


configure_lora_weights(os.environ.get("VLM_LORA_WEIGHTS") or VLM_LORA_WEIGHTS)


def _model_input_device(model: Any, torch: Any) -> Any:
    if torch.cuda.is_available():
        return "cuda"
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _model_input_dtype(model: Any, torch: Any) -> Any:
    try:
        dtype = next(model.parameters()).dtype
        if getattr(dtype, "is_floating_point", False):
            return dtype
    except StopIteration:
        pass
    return torch.float32


def _build_single_image_messages(
    query_image: str | Path,
    prompt: str,
) -> list[dict[str, Any]]:
    image_path = _resolve_query_image_path(query_image)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _prepare_gemma3_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    gemma_messages: list[dict[str, Any]] = []
    images: list[Any] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            gemma_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
            continue

        gemma_content: list[dict[str, Any]] = []
        for item in content:
            item_type = item.get("type")
            if item_type == "image":
                from PIL import Image

                image = Image.open(_resolve_image_path(item["image"])).convert("RGB")
                images.append(image)
                gemma_content.append({"type": "image", "image": image})
            elif item_type == "text":
                gemma_content.append({
                    "type": "text",
                    "text": str(item.get("text", "")),
                })

        gemma_messages.append({"role": role, "content": gemma_content})

    return gemma_messages, images


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts = [
        str(item.get("text", "")).strip()
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part)


def _prepare_internvl_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[Any]]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    images: list[Any] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            text = _message_text(content)
            if text:
                system_parts.append(text)
            continue

        if isinstance(content, str):
            text = content.strip()
            if text:
                user_parts.append(text)
            continue

        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                from PIL import Image

                images.append(
                    Image.open(_resolve_image_path(item["image"])).convert("RGB")
                )
                if len(images) == 1:
                    user_parts.append("<image>")
                else:
                    user_parts.append(f"Image {len(images)}: <image>")
            elif item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    user_parts.append(text)

    system_text = "\n\n".join(system_parts).strip()
    user_text = "\n".join(user_parts).strip()
    if system_text and user_text:
        return f"{system_text}\n\n{user_text}", images
    return system_text or user_text, images


def _run_qwen25vl_messages(
    messages: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    model, processor, _, process_vision_info, torch = _vlm_components()

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model_input_device(model, torch))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def _run_gemma3_messages(
    messages: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    model, processor, _, _, torch = _vlm_components()
    gemma_messages, images = _prepare_gemma3_messages(messages)
    prompt_text = processor.apply_chat_template(
        gemma_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[prompt_text],
        images=[images] if images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model_input_device(model, torch))

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "top_p": None,
        "top_k": None,
    }
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(**inputs, **generation_kwargs)
        else:
            generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = int(inputs["attention_mask"][0].sum().item())
    generated = generated_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _run_internvl_messages(
    messages: list[dict[str, Any]],
    max_new_tokens: int,
) -> str:
    model, tokenizer, _, transform, torch = _vlm_components()
    question, images = _prepare_internvl_messages(messages)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    device = _model_input_device(model, torch)

    pixel_values = None
    num_patches_list = None
    if images:
        dtype = _model_input_dtype(model, torch)
        pixel_values = torch.cat(
            [transform(image).unsqueeze(0).to(dtype) for image in images],
            dim=0,
        ).to(device)
        num_patches_list = [1] * len(images)

    chat_kwargs = {
        "tokenizer": tokenizer,
        "pixel_values": pixel_values,
        "question": question,
        "generation_config": generation_config,
    }
    if num_patches_list is not None:
        chat_kwargs["num_patches_list"] = num_patches_list

    with torch.inference_mode():
        return model.chat(**chat_kwargs).strip()


def _run_vlm(
    query_image: str | Path,
    prompt: str,
    max_new_tokens: int = VLM_MAX_NEW_TOKENS,
) -> str:
    return _run_vlm_messages(
        _build_single_image_messages(query_image, prompt),
        max_new_tokens=max_new_tokens,
    )


def _run_vlm_messages(
    messages: list[dict[str, Any]],
    max_new_tokens: int = VLM_MAX_NEW_TOKENS,
) -> str:
    """Run the configured VLM with a pre-built messages list."""
    _, _, backend, _, _ = _vlm_components()
    if backend == GEMMA3_BACKEND:
        return _run_gemma3_messages(messages, max_new_tokens=max_new_tokens)
    if backend == INTERNVL_BACKEND:
        return _run_internvl_messages(messages, max_new_tokens=max_new_tokens)
    return _run_qwen25vl_messages(messages, max_new_tokens=max_new_tokens)


def preload_vlm_model() -> None:
    """Load only the generation model before serving baseline requests."""
    _vlm_components()


def preload_models() -> None:
    """Load both retrieval and generation models before serving RAG requests."""
    from embedding import get_embedding_image_processor, get_embedding_model

    get_embedding_image_processor()
    get_embedding_model()
    preload_vlm_model()


def build_baseline_prompt(task_type: str, query: str | None = None) -> str:
    task_type = _validate_task_type(task_type)
    query = query or _default_query_for_task(task_type)
    if task_type == CONSTRUCTIONSITE10K_TASK:
        from rag_answer import CONSTRUCTIONSITE10K_SYSTEM_PROMPT

        return f"""{CONSTRUCTIONSITE10K_SYSTEM_PROMPT}

Question for the query image:
{query}
"""
    if task_type == LAB_SAFETY_TASK:
        from rag_answer import LAB_SAFETY_SYSTEM_PROMPT

        return f"""{LAB_SAFETY_SYSTEM_PROMPT}

Question for the query image:
{query}
"""
    if task_type == LAB_SAFETY_GEN_TASK:
        from rag_answer import LAB_SAFETY_GEN_SYSTEM_PROMPT

        return f"""{LAB_SAFETY_GEN_SYSTEM_PROMPT}

Question for the query image:
{query}
"""
    if task_type == SAFETY_LEVEL_TASK:
        from rag_answer import INSPECSAFE_SAFETY_LEVEL_SYSTEM_PROMPT

        return f"""{INSPECSAFE_SAFETY_LEVEL_SYSTEM_PROMPT}

Question for the query image:
{query}
"""

    return f"""You are a construction safety visual inspection assistant.

Question for the query image:
{query}

Use only the query image as evidence.

Return your answer in this format:
Query image observations:
Reasoning:
Final label: safe or unsafe
"""


def VLM_inference(
    task_type: str,
    query_image: str | Path,
    *,
    query: str | None = None,
    max_new_tokens: int = VLM_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run baseline VLM inference without retrieval context."""
    task_type = _validate_task_type(task_type)
    query = query or _default_query_for_task(task_type)
    prompt = build_baseline_prompt(task_type, query)
    output = _run_vlm(query_image, prompt, max_new_tokens=max_new_tokens)
    return {
        "task_type": task_type,
        "query_image": str(_resolve_query_image_path(query_image)),
        "query": query,
        "prompt": prompt,
        "output": output,
    }


def VLM_inference_two_stage(
    task_type: str,
    query_image: str | Path,
    *,
    query: str | None = None,
    stage_one_max_new_tokens: int = INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    stage_two_max_new_tokens: int = INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run gated two-stage InspecSafe classification without retrieval."""
    from two_stage_inference import run_two_stage_safety_inference

    task_type = _validate_task_type(task_type)
    if task_type != SAFETY_JUDGEMENT_TASK:
        raise ValueError(
            "Two-stage inference currently supports only 'safety judgement'."
        )

    query = query or _default_query_for_task(task_type)
    image_path = _resolve_query_image_path(query_image)
    result = run_two_stage_safety_inference(
        image_path,
        query,
        _run_vlm,
        stage_one_max_new_tokens=stage_one_max_new_tokens,
        stage_two_max_new_tokens=stage_two_max_new_tokens,
    )
    prompts = {
        "stage_one": result["stage_one"]["prompt"],
        "stage_two": (
            result["stage_two"]["prompt"] if result["stage_two"] else None
        ),
    }
    return {
        "task_type": task_type,
        "query_image": str(image_path),
        "query": query,
        "prompt": prompts,
        **result,
    }


def VLM_inference_with_RAG(
    task_type: str,
    query_image: str | Path,
    *,
    query: str | None = None,
    top_k: int = TOP_K,
    gated_rag: float = GATED_RAG,
    max_new_tokens: int = VLM_MAX_NEW_TOKENS,
    debug_mode: bool = False
) -> dict[str, Any]:
    """Retrieve similar examples, build a RAG prompt, and run the configured VLM."""
    from rag_answer import (
        build_constructionsite10k_rag_messages,
        build_inspecsafe_safety_level_rag_messages,
        build_labsafety_gen_rag_messages,
        build_labsafety_rag_messages,
        build_rag_messages,
    )
    from retrieval_gating import gate_retrieval_results
    from retriever import search_by_query_image

    task_type = _validate_task_type(task_type)
    query = query or _default_query_for_task(task_type)
    image_path = _resolve_query_image_path(query_image)
    top_k_retrieved = search_by_query_image(
        query_image,
        top_k=top_k,
        dataset=TASK_TO_RAG_DATASET[task_type],
    )
    retrieved = gate_retrieval_results(top_k_retrieved, gated_rag)
    if debug_mode:
        from retriever import copy_image_to_demo, save_retrieved_images

        print(f"images saved for debug_mode")
        save_retrieved_images(retrieved)
        copy_image_to_demo(image_path, "query_image.png")
    if task_type == CONSTRUCTIONSITE10K_TASK:
        messages = build_constructionsite10k_rag_messages(query, image_path, retrieved)
    elif task_type == LAB_SAFETY_TASK:
        messages = build_labsafety_rag_messages(query, image_path, retrieved)
    elif task_type == LAB_SAFETY_GEN_TASK:
        messages = build_labsafety_gen_rag_messages(query, image_path, retrieved)
    elif task_type == SAFETY_LEVEL_TASK:
        messages = build_inspecsafe_safety_level_rag_messages(
            query, image_path, retrieved
        )
    else:
        messages = build_rag_messages(query, image_path, retrieved)
    output = _run_vlm_messages(messages, max_new_tokens=max_new_tokens)
    return {
        "task_type": task_type,
        "query_image": str(image_path),
        "query": query,
        "top_k": top_k,
        "gated_rag": float(gated_rag),
        "retrieved_count_before_gate": len(top_k_retrieved),
        "retrieved_count": len(retrieved),
        "retrieved": retrieved,
        "prompt": messages,
        "output": output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM safety inference.")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "InspecSafe" / "dataset.csv",
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline inference without RAG context.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gated-rag",
        "--gated_rag",
        dest="gated_rag",
        type=float,
        default=GATED_RAG,
        help="Keep top-k results with cosine similarity >= this threshold.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=VLM_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run.")
    parser.add_argument("--offset", type=int, default=0, help="Samples to skip.")
    add_lora_cli_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import pandas as pd
    from evaluate_inspecsafe import extract_label

    args = parse_args()
    if args.lora_weights is not None:
        configure_lora_weights(args.lora_weights)
    df = pd.read_csv(args.dataset_csv)
    df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.iloc[: args.limit]

    mode = "baseline" if args.baseline else "rag"
    total = len(df)
    correct = 0
    evaluated = 0

    print(
        f"Mode: {mode} | Samples: {total} | top_k: {args.top_k} "
        f"| gated_rag: {args.gated_rag}"
    )
    print("-" * 60)

    for _, row in df.iterrows():
        sample_id = row["id"]
        image_path = row["image_path"]
        ground_truth = str(row["safe_label"]).strip().lower()

        try:
            if args.baseline:
                result = VLM_inference(
                    "safety judgement", image_path,
                    max_new_tokens=args.max_new_tokens
                )
            else:
                result = VLM_inference_with_RAG(
                    "safety judgement", image_path,
                    top_k=args.top_k,
                    gated_rag=args.gated_rag,
                    max_new_tokens=args.max_new_tokens,
                    debug_mode=True
                )

            predicted = extract_label(result["output"])
            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1
            evaluated += 1

            tag = "OK" if is_correct else "WRONG"
            print(f"[{sample_id}] {tag} | truth={ground_truth} pred={predicted}")
            print(f"  Output: {result['output'][:120]}")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[{sample_id}] ERROR - {exc}")

    print("-" * 60)
    accuracy = correct / evaluated if evaluated > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{evaluated})")
