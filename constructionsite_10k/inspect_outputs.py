"""
查看模型原始输出脚本
支持 Gemma3 / Qwen2.5-VL / InternVL2
随机抽样若干条，打印原始输出、parse 结果、GT 对比
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import random
import torch
from pathlib import Path
from PIL import Image

# ==================== 配置 ====================
#MODEL_TYPE  = "gemma3"    # 改成 "qwen25vl" 或 "internvl2"
#MODEL_TYPE = "qwen25vl"
MODEL_TYPE = "internvl2"
NUM_SAMPLES = 20          # 查看几条
RANDOM_SEED = 42
MAX_NEW_TOKENS = 512

MODEL_PATHS = {
    "gemma3":   "/root/autodl-tmp/gemma-3-4b-it",
    "qwen25vl": "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct",
    "internvl2": "/root/autodl-tmp/InternVL2-4B",
}
TEST_JSON  = "/root/autodl-tmp/test.json"
IMAGE_ROOT = "/root/autodl-tmp/images"

SYSTEM_PROMPT = """You are a professional construction site safety inspector with expertise in hazard identification and regulatory compliance.

Carefully analyze the provided construction site image and assess safety compliance step by step.

## Safety Rules

Rule 1 - Personal Protective Equipment (PPE):
All workers on foot must wear: hard hats, clothes covering shoulders and legs, toe-covering shoes. When cutting/welding/grinding/drilling: face shields or safety glasses. At night: high-visibility retroreflective vests.

Rule 2 - Working at Height:
Workers at heights >= 3 meters with unprotected edges must wear a safety harness.

Rule 3 - Edge Protection:
Underground excavations >= 3 meters deep with steep retaining walls require guardrails or warning fences when workers are present.

Rule 4 - Excavator Proximity:
No worker shall appear in the blind spots or within the operation radius of an active excavator, or any excavator with an operator inside.

## Instructions

Step 1 - Scene Description: Describe what you observe including workers, positions, activities, equipment, and environment.
Step 2 - Rule Analysis: For each rule, state whether it is complied with or violated with specific visual evidence.
Step 3 - Output the following JSON only, no extra text:

{
  "annotation": "<detailed scene description>",
  "violations": [
    {
      "rule": <rule_id as integer>,
      "reason": "<specific visual evidence>"
    }
  ]
}

If no violations are found, return an empty list for violations."""

MODEL_NAME = MODEL_PATHS[MODEL_TYPE]

# ==================== 工具函数 ====================
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        sample["image"] = sample["image"].replace("\\", "/")
    return data

def parse_json_output(text):
    try:
        result = json.loads(text.strip())
        if "violations" in result:
            return result, True
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if "violations" in result:
                    return result, True
            except:
                pass
    return {"annotation": "", "violations": []}, False

def get_gt_info(sample):
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            gt_json, _ = parse_json_output(msg["content"])
            return gt_json
    return {}

# ==================== 加载模型 ====================
print(f"🚀 加载 {MODEL_TYPE} 模型...")
device = torch.device("cuda:0")

if MODEL_TYPE == "gemma3":
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

elif MODEL_TYPE == "qwen25vl":
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
    from qwen_vl_utils import process_vision_info
    config    = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=512*28*28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, config=config, device_map={"": "cuda:0"},
        torch_dtype=torch.float16, trust_remote_code=True,
    )

elif MODEL_TYPE == "internvl2":
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer
    IMG_SIZE = 448
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    NUM_VISUAL_TOKENS = 256
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map={"": "cuda:0"},
        torch_dtype=torch.float16, trust_remote_code=True,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

model.eval()
print("✅ 模型加载完成")

# ==================== 抽样并推理 ====================
raw_test = load_data(TEST_JSON)
random.seed(RANDOM_SEED)
samples  = random.sample(raw_test, min(NUM_SAMPLES, len(raw_test)))

parse_success = 0
parse_fail    = 0

for idx, sample in enumerate(samples):
    # 提取 user_text 和 img_path
    user_text = ""
    img_path  = None
    for msg in sample["messages"]:
        if msg["role"] == "user":
            for item in msg["content"]:
                if item["type"] == "image":
                    img_path = Path(IMAGE_ROOT) / Path(sample["image"]).name
                elif item["type"] == "text":
                    user_text = item["text"].strip()

    image = Image.open(img_path).convert("RGB")

    # 推理
    if MODEL_TYPE == "gemma3":
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        prompt_text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=prompt_text, images=[image],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, top_p=None, top_k=None,
                )
        real_len  = inputs["attention_mask"][0].sum().item()
        generated = output_ids[0][real_len:]
        pred_text = processor.decode(generated, skip_special_tokens=True).strip()

    elif MODEL_TYPE == "qwen25vl":
        conv = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conv)
        inputs = processor(
            text=[text], images=image_inputs,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                )
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        pred_text = processor.decode(generated, skip_special_tokens=True).strip()

    elif MODEL_TYPE == "internvl2":
        pixel_values = transform(image).unsqueeze(0).to(torch.float16).to(device)
        question = f"{SYSTEM_PROMPT}\n\n{user_text}"
        with torch.no_grad():
            pred_text = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False),
            )

    # parse 结果
    pred_json, parse_ok = parse_json_output(pred_text)
    gt_info = get_gt_info(sample)

    if parse_ok:
        parse_success += 1
        status = "✅ parse 成功"
    else:
        parse_fail += 1
        status = "❌ parse 失败"

    # 打印
    print(f"\n{'='*70}")
    print(f"样本 {idx+1}/{NUM_SAMPLES}  [{status}]  图片: {sample['image']}")
    print(f"{'='*70}")
    print(f"\n[GT violations]: {[v['rule'] for v in gt_info.get('violations', [])]}")
    if parse_ok:
        print(f"[Pred violations]: {[v['rule'] for v in pred_json.get('violations', [])]}")
    print(f"\n[模型原始输出] ({len(pred_text)} chars):")
    print("-" * 40)
    print(pred_text[:800])   # 只打印前 800 字符
    if len(pred_text) > 800:
        print(f"... (truncated, total {len(pred_text)} chars)")

# ==================== 汇总 ====================
print(f"\n{'='*70}")
print(f"📊 汇总  总样本: {NUM_SAMPLES}")
print(f"  ✅ parse 成功: {parse_success} ({parse_success/NUM_SAMPLES*100:.0f}%)")
print(f"  ❌ parse 失败: {parse_fail}    ({parse_fail/NUM_SAMPLES*100:.0f}%)")
