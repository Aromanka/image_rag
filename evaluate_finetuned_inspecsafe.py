"""Evaluate the configured Qwen2.5-VL or Gemma 3 model on InspecSafe-V1.

The model, prompt, generation, metrics, and resolution sweep follow
``.plan/reference/qwen25vl_eval_finetuned_inspecsafe.py``.  Dataset loading
and image-path resolution follow ``evaluate_inspecsafe_safety_level.py`` so
the evaluator can read this repository's pipeline JSON without requiring a
separate, flattened ``pipeline_images`` directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from config import (
    INSPECSAFE_DATA_ROOT,
    PROJECT_ROOT,
    SBERT_MODEL_PATH,
    VLM_LORA_WEIGHTS,
    VLM_MODEL_PATH,
    VLM_PROCESSOR_PATH,
    VLM_USE_FLASH_ATTENTION,
)
from evaluate_inspecsafe_safety_level import (
    assistant_label,
    load_inspecsafe_safety_level_data,
    print_safety_level_results,
    resolve_sample_image,
)
from utils.evaluate_utils import evaluate_inspecsafe_safety_level_results_json


TEST_JSON = PROJECT_ROOT / "data" / "inspecsafe_pipeline" / "pipeline_test.json"
DATA_ROOT = Path(INSPECSAFE_DATA_ROOT)
SBERT_PATH = SBERT_MODEL_PATH
EVAL_OUTPUT = PROJECT_ROOT / "save" / "eval_finetuned_inspecsafe_resolution.json"
MAX_TEST_SAMPLES = None
EVAL_BATCH_SIZE = 8
PARSE_FAIL_DIR = PROJECT_ROOT / "save" / "parse_failures" / "inspecsafe"

# None means the original resolution; other values are square pixel sizes.
IMG_SIZES = [None, 224, 336, 448, 560]

SYSTEM_PROMPT = """You are an industrial safety inspector reviewing footage from an autonomous inspection robot deployed at an oil and gas / petrochemical facility.

Carefully analyse the provided image and assess the safety situation step by step. Industrial inspection scenes frequently contain hazards; examine ALL personnel, equipment, and the surrounding environment. Do NOT assume the scene is safe without thorough inspection.

## Safety Level Criteria (Oil & Gas / Chemical)

Level I  - CRITICAL (most severe, immediate threat):
    Open flame, smoke, worker not wearing hard hat, worker not wearing gloves,
    worker not wearing mask, smoking, personnel collapse, oil leakage.

Level II - MODERATE hazard:
    Water pooling, worker using a mobile phone.

Level III - MINOR / potential risk:
    Foreign objects present in the area.

Level IV - NORMAL:
    No safety factors observed.

When multiple hazards co-exist, the overall safety level is determined by the MOST severe hazard present.

## Instructions

Step 1 - Scene Description: Describe the scene, key equipment, personnel, and observed conditions.
Step 2 - Hazard Analysis: Identify each hazard present using the canonical phrases below.
Step 3 - Output the following JSON only, no extra text:

{
  "scene_description": "<detailed scene description>",
  "hazards": ["<hazard 1>", "<hazard 2>"],
  "overall_safety_level": "<Level I | Level II | Level III | Level IV>"
}

Use exactly these canonical hazard phrases when applicable: "open flame", "smoke",
"smoking", "no hard hat", "no gloves", "no mask", "personnel collapse", "oil leakage",
"water pooling", "using mobile phone", "foreign objects".
If no safety factors are present, return an empty hazards list and "Level IV"."""


def _model_backend() -> str:
    model_hint = f"{VLM_MODEL_PATH} {VLM_PROCESSOR_PATH}".lower()
    return "gemma3" if "gemma" in model_hint else "qwen2_5_vl"


def _lora_path() -> Path | None:
    if not str(VLM_LORA_WEIGHTS).strip():
        return None
    path = Path(VLM_LORA_WEIGHTS).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _processor_path(lora_path: Path | None) -> str | Path:
    if lora_path is None or not lora_path.is_dir():
        return VLM_PROCESSOR_PATH
    markers = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    return lora_path if any((lora_path / marker).exists() for marker in markers) else VLM_PROCESSOR_PATH


def load_model():
    """Load the model, processor, and optional LoRA configured in config.py."""
    backend = _model_backend()
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if backend == "gemma3":
        from transformers import Gemma3ForConditionalGeneration

        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        model_class = Gemma3ForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_kwargs["torch_dtype"] = "auto"
        model_class = Qwen2_5_VLForConditionalGeneration

    if VLM_USE_FLASH_ATTENTION:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = model_class.from_pretrained(VLM_MODEL_PATH, **model_kwargs)
    lora_path = _lora_path()
    if lora_path is not None:
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights path not found: {lora_path}")
        model = PeftModel.from_pretrained(
            model,
            str(lora_path),
            is_trainable=False,
        )
        print(f"Loaded LoRA weights: {lora_path}")
    else:
        print("Not using LoRA weights.")

    processor = AutoProcessor.from_pretrained(
        str(_processor_path(lora_path)),
        trust_remote_code=True,
    )
    model.eval()
    return model, processor, backend


def load_test_data(
    dataset_json: Path,
    data_root: Path,
    max_samples: int | None,
) -> list[dict]:
    """Load pipeline samples and attach their resolved original image paths."""
    samples = load_inspecsafe_safety_level_data(dataset_json)
    if max_samples is not None:
        samples = samples[:max_samples]

    loaded = []
    for sample in samples:
        normalized = dict(sample)
        normalized["image"] = str(sample["image"]).replace("\\", "/")
        normalized["_resolved_image"] = str(
            resolve_sample_image(
                normalized,
                dataset_json=dataset_json,
                image_root=None,
                data_root=data_root,
            )
        )
        loaded.append(normalized)
    return loaded


def _report_parse_failures(records: list[dict], parse_fail_path: str, desc: str) -> None:
    """Report and archive parse failures using resolved dataset image paths."""
    import zipfile

    failed = [record for record in records if record.get("parse_failed")]
    print(f"\n  [Parse failures: {len(failed)}]  (desc={desc!r})")
    if not failed:
        return

    for index, record in enumerate(failed, start=1):
        print(f"\n  -- failure {index}/{len(failed)} --")
        print(f"  image    : {Path(record['image']).name}")
        print(f"  gt_level : {record['gt_level']}")
        print(f"  raw_output (first 300 chars): {record['output'][:300]!r}")

    zip_path = Path(parse_fail_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for record in failed:
            source = Path(record["resolved_image"])
            if source.exists():
                archive.write(source, arcname=source.name)
    print(f"\n  images saved -> {zip_path}")


@torch.no_grad()
def evaluate_model(
    model,
    processor,
    backend,
    test_data,
    system_prompt,
    sbert_path=None,
    desc="eval",
    eval_batch_size=8,
    max_new_tokens=384,
    parse_fail_dir=None,
    img_size=None,
):
    """Run reference-style batched inference for Qwen2.5-VL or Gemma 3."""
    process_vision_info = None
    if backend == "qwen2_5_vl":
        from qwen_vl_utils import process_vision_info

    model.eval()
    processor.tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    prepared = []
    for sample in test_data:
        gt_label = assistant_label(sample) or {}
        image = Image.open(sample["_resolved_image"]).convert("RGB")
        if img_size is not None:
            image = image.resize((img_size, img_size), Image.BILINEAR)
        user_text = next(
            (
                item["text"]
                for message in sample["messages"]
                if message["role"] == "user"
                for item in message["content"]
                if item.get("type") == "text"
            ),
            "",
        )

        system_content = (
            system_prompt
            if backend == "qwen2_5_vl"
            else [{"type": "text", "text": system_prompt}]
        )
        conversation = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        if backend == "qwen2_5_vl":
            image_inputs, _ = process_vision_info(conversation)
        else:
            image_inputs = image
        prepared.append(
            {
                "sample": sample,
                "gt_label": gt_label,
                "text": text,
                "image_inputs": image_inputs,
            }
        )

    records = []
    for start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        items = prepared[start : start + eval_batch_size]
        inputs = processor(
            text=[item["text"] for item in items],
            images=[item["image_inputs"] for item in items],
            return_tensors="pt",
            padding=True,
        ).to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        for index, item in enumerate(items):
            real_len = inputs["attention_mask"][index].sum().item()
            output_ids = generated[index][real_len:]
            output_text = processor.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            records.append(
                {
                    "image": item["sample"]["image"],
                    "resolved_image": item["sample"]["_resolved_image"],
                    "ground_truth": item["gt_label"],
                    "output": output_text,
                }
            )

    evaluated = evaluate_inspecsafe_safety_level_results_json(
        {"results": records},
        compute_scene_metrics=True,
        sbert_path=sbert_path,
    )
    if parse_fail_dir:
        _report_parse_failures(records, parse_fail_dir, desc)
    return evaluated


def main() -> None:
    raw_test = load_test_data(TEST_JSON, DATA_ROOT, MAX_TEST_SAMPLES)
    print(f"test: {len(raw_test)}")

    print(f"Loading configured model: {VLM_MODEL_PATH}")
    model, processor, backend = load_model()
    print(f"  Backend: {backend}")
    print(f"  Processor: {_processor_path(_lora_path())}")
    print(f"  GPU: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    all_results = {}
    for img_size in IMG_SIZES:
        size_label = str(img_size) if img_size is not None else "original"
        print(f"\n{'=' * 60}")
        print(f"  Resolution: {size_label}")
        print(f"{'=' * 60}")

        evaluated = evaluate_model(
            model,
            processor,
            backend,
            raw_test,
            system_prompt=SYSTEM_PROMPT,
            sbert_path=SBERT_PATH,
            desc=f"{backend} InspecSafe [{size_label}]",
            eval_batch_size=EVAL_BATCH_SIZE,
            parse_fail_dir=f"{PARSE_FAIL_DIR}_{size_label}.zip",
            img_size=img_size,
        )
        print_safety_level_results(evaluated["summary"])
        all_results[size_label] = evaluated["summary"]

    output_path = Path(EVAL_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)
    print(f"\nAll resolution results saved to {output_path}")


if __name__ == "__main__":
    main()
