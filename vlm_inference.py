"""VLM inference entry points with optional image RAG context."""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import (
    CONSTRUCTIONSITE10K_TASK,
    DEFAULT_SAFETY_QUERY,
    DEFAULT_CONSTRUCTIONSITE10K_QUERY,
    PROJECT_ROOT,
    SUPPORTED_TASK_TYPES,
    TOP_K,
    VLM_MAX_NEW_TOKENS,
    VLM_MODEL_PATH,
    VLM_PROCESSOR_PATH,
    VLM_USE_FLASH_ATTENTION,
)


QWEN25VL_BACKEND = "qwen2_5_vl"
GEMMA3_BACKEND = "gemma3"


def _validate_task_type(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized not in SUPPORTED_TASK_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise ValueError(f"Unsupported task_type '{task_type}'. Supported: {supported}.")
    return normalized


def _default_query_for_task(task_type: str) -> str:
    task_type = _validate_task_type(task_type)
    if task_type == CONSTRUCTIONSITE10K_TASK:
        return DEFAULT_CONSTRUCTIONSITE10K_QUERY
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
    return QWEN25VL_BACKEND


@lru_cache(maxsize=1)
def _vlm_components() -> tuple[Any, Any, str, Any, Any]:
    import torch
    from transformers import AutoProcessor

    backend = _infer_vlm_backend()

    if backend == GEMMA3_BACKEND:
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
        processor = AutoProcessor.from_pretrained(
            VLM_PROCESSOR_PATH,
            trust_remote_code=True,
        )
        model.eval()
        return model, processor, backend, None, torch

    from qwen_vl_utils import process_vision_info
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
    processor = AutoProcessor.from_pretrained(VLM_PROCESSOR_PATH)
    model.eval()
    return model, processor, backend, process_vision_info, torch


def _model_input_device(model: Any, torch: Any) -> Any:
    if torch.cuda.is_available():
        return "cuda"
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


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
    return _run_qwen25vl_messages(messages, max_new_tokens=max_new_tokens)


def build_baseline_prompt(task_type: str, query: str | None = None) -> str:
    task_type = _validate_task_type(task_type)
    query = query or _default_query_for_task(task_type)
    if task_type == CONSTRUCTIONSITE10K_TASK:
        from rag_answer import CONSTRUCTIONSITE10K_SYSTEM_PROMPT

        return f"""{CONSTRUCTIONSITE10K_SYSTEM_PROMPT}

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


def VLM_inference_with_RAG(
    task_type: str,
    query_image: str | Path,
    *,
    query: str | None = None,
    top_k: int = TOP_K,
    max_new_tokens: int = VLM_MAX_NEW_TOKENS,
    debug_mode: bool = False
) -> dict[str, Any]:
    """Retrieve similar examples, build a RAG prompt, and run the configured VLM."""
    from rag_answer import build_constructionsite10k_rag_messages, build_rag_messages
    from retriever import search_by_query_image

    task_type = _validate_task_type(task_type)
    query = query or _default_query_for_task(task_type)
    image_path = _resolve_query_image_path(query_image)
    retrieved = search_by_query_image(query_image, top_k=top_k)
    if debug_mode:
        from retriever import copy_image_to_demo, save_retrieved_images

        print(f"images saved for debug_mode")
        save_retrieved_images(retrieved)
        copy_image_to_demo(image_path, "query_image.png")
    if task_type == CONSTRUCTIONSITE10K_TASK:
        messages = build_constructionsite10k_rag_messages(query, image_path, retrieved)
    else:
        messages = build_rag_messages(query, image_path, retrieved)
    output = _run_vlm_messages(messages, max_new_tokens=max_new_tokens)
    return {
        "task_type": task_type,
        "query_image": str(image_path),
        "query": query,
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
    parser.add_argument("--max-new-tokens", type=int, default=VLM_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run.")
    parser.add_argument("--offset", type=int, default=0, help="Samples to skip.")
    return parser.parse_args()


if __name__ == "__main__":
    import pandas as pd
    from evaluate_inspecsafe import extract_label

    args = parse_args()
    df = pd.read_csv(args.dataset_csv)
    df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.iloc[: args.limit]

    mode = "baseline" if args.baseline else "rag"
    total = len(df)
    correct = 0
    evaluated = 0

    print(f"Mode: {mode} | Samples: {total} | top_k: {args.top_k}")
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
