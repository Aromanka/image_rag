"""Evaluate the fine-tuned Qwen2.5-VL LoRA on InspecSafe-V1.

The model, prompt, generation, metrics, and resolution sweep follow
``.plan/reference/qwen25vl_eval_finetuned_inspecsafe.py``.  Dataset loading
and image-path resolution follow ``evaluate_inspecsafe_safety_level.py`` so
the evaluator can read this repository's pipeline JSON without requiring a
separate, flattened ``pipeline_images`` directory.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import INSPECSAFE_DATA_ROOT, PROJECT_ROOT
from evaluate_inspecsafe_safety_level import (
    load_inspecsafe_safety_level_data,
    resolve_sample_image,
)

# Reuse the reference metric implementation so results remain directly
# comparable with the original experiment.
REFERENCE_DIR = PROJECT_ROOT / ".plan" / "reference"
sys.path.insert(0, str(REFERENCE_DIR))
from inspecsafe_eval_utils import (  # noqa: E402
    LEVELS,
    _accumulate,
    _finalise,
    _load_sbert,
    assistant_label,
    print_results,
)


MODEL_NAME = "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct"
LORA_DIR = "/root/autodl-tmp/qwen25vl_3b_lora_inspecsafe"
USE_SYNTHETIC = 1  # 0 = real data only, 1 = real + synthetic
TEST_JSON = PROJECT_ROOT / "data" / "inspecsafe_pipeline" / "pipeline_test.json"
DATA_ROOT = Path(INSPECSAFE_DATA_ROOT)
SBERT_PATH = "/root/autodl-tmp/all-MiniLM-L6-v2"
EVAL_OUTPUT = "/root/autodl-tmp/eval_qwen25vl_inspecsafe_resolution.json"
MAX_TEST_SAMPLES = None
EVAL_BATCH_SIZE = 8
PARSE_FAIL_DIR = "/root/autodl-tmp/parse_failures/qwen25vl"

if USE_SYNTHETIC:
    LORA_DIR += "_syn"
    PARSE_FAIL_DIR += "_syn"

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


def load_model():
    """Load Qwen2.5-VL exactly as in the reference evaluator."""
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        config=config,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    return model, processor


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    print(f"  GPU freed: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


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

    failed = [record for record in records if record["pred"].get("_raw") is not None]
    print(f"\n  [Parse failures: {len(failed)}]  (desc={desc!r})")
    if not failed:
        return

    for index, record in enumerate(failed, start=1):
        print(f"\n  -- failure {index}/{len(failed)} --")
        print(f"  image    : {Path(record['image']).name}")
        print(f"  gt_level : {record['gt_level']}")
        print(f"  raw_output (first 300 chars): {record['pred']['_raw'][:300]!r}")

    zip_path = Path(parse_fail_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for record in failed:
            source = Path(record["resolved_image"])
            if source.exists():
                archive.write(source, arcname=source.name)
    print(f"\n  images saved -> {zip_path}")


@torch.no_grad()
def evaluate_qwen25vl(
    model,
    processor,
    test_data,
    system_prompt,
    sbert_path=None,
    desc="eval",
    eval_batch_size=8,
    max_new_tokens=384,
    parse_fail_dir=None,
    img_size=None,
):
    """Reference Qwen evaluator with repository-aware image resolution."""
    from qwen_vl_utils import process_vision_info

    model.eval()
    processor.tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    sbert = _load_sbert(sbert_path, device)

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

        conversation = [
            {"role": "system", "content": system_prompt},
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
        image_inputs, _ = process_vision_info(conversation)
        prepared.append(
            {
                "sample": sample,
                "gt_label": gt_label,
                "text": text,
                "image_inputs": image_inputs,
            }
        )

    records = []
    n_parse_ok = level_correct = 0
    tp = fp = fn = 0
    gt_descs, pred_descs = [], []
    lvl_tp = {level: 0 for level in LEVELS}
    lvl_fp = {level: 0 for level in LEVELS}
    lvl_fn = {level: 0 for level in LEVELS}

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
            n_parse_ok, level_correct, tp, fp, fn = _accumulate(
                records,
                item["sample"],
                item["gt_label"],
                output_text,
                n_parse_ok,
                level_correct,
                tp,
                fp,
                fn,
                gt_descs,
                pred_descs,
                lvl_tp,
                lvl_fp,
                lvl_fn,
            )
            records[-1]["resolved_image"] = item["sample"]["_resolved_image"]

    result = _finalise(
        records,
        len(test_data),
        n_parse_ok,
        level_correct,
        tp,
        fp,
        fn,
        gt_descs,
        pred_descs,
        sbert,
        device,
        lvl_tp,
        lvl_fp,
        lvl_fn,
    )
    if parse_fail_dir:
        _report_parse_failures(records, parse_fail_dir, desc)
    return result


def main() -> None:
    raw_test = load_test_data(TEST_JSON, DATA_ROOT, MAX_TEST_SAMPLES)
    print(f"test: {len(raw_test)}")

    print("Loading Qwen2.5-VL LoRA model (loaded once)...")
    model, processor = load_model()
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    print(f"  GPU: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    all_results = {}
    for img_size in IMG_SIZES:
        size_label = str(img_size) if img_size is not None else "original"
        print(f"\n{'=' * 60}")
        print(f"  Resolution: {size_label}")
        print(f"{'=' * 60}")

        result = evaluate_qwen25vl(
            model,
            processor,
            raw_test,
            system_prompt=SYSTEM_PROMPT,
            sbert_path=SBERT_PATH,
            desc=f"Qwen InspecSafe [{size_label}]",
            eval_batch_size=EVAL_BATCH_SIZE,
            parse_fail_dir=f"{PARSE_FAIL_DIR}_{size_label}.zip",
            img_size=img_size,
        )
        print_results(f"Qwen2.5-VL LoRA InspecSafe - {size_label}", result)
        all_results[size_label] = {
            key: value for key, value in result.items() if key != "records"
        }

    output_path = Path(EVAL_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)
    print(f"\nAll resolution results saved to {output_path}")


if __name__ == "__main__":
    main()
