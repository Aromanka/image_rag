"""
独立评估脚本：加载微调后的 InternVL2-4B LoRA 模型，在 test.json 上评估
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from evaluate_utils import evaluate_internvl2, print_results

# ==================== 0. 配置 ====================
MODEL_NAME        = "/root/autodl-tmp/InternVL2-4B"
LORA_DIR          = "/root/autodl-tmp/internvl2_4b_lora"
TEST_JSON         = "/root/autodl-tmp/test.json"
IMAGE_ROOT        = "/root/autodl-tmp/images"
SBERT_PATH        = "/root/autodl-tmp/all-MiniLM-L6-v2"
EVAL_OUTPUT       = "/root/autodl-tmp/eval_internvl2_finetuned.json"
MAX_TEST_SAMPLES  = None
EVAL_BATCH_SIZE   = 8
IMG_SIZE          = 448
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
NUM_VISUAL_TOKENS = 256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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

# ==================== 图像预处理 ====================
transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ==================== 1. 加载数据 ====================
def load_data(json_path, max_samples=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        sample["image"] = sample["image"].replace("\\", "/")
    if max_samples:
        data = data[:max_samples]
    return data

raw_test = load_data(TEST_JSON, MAX_TEST_SAMPLES)
print(f"✅ 测试集: {len(raw_test)} 条")

# ==================== 2. 加载微调模型 ====================
print("🚀 加载微调后的 InternVL2-4B 模型...")
tokenizer  = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, use_fast=False,
)
base_model = AutoModel.from_pretrained(
    MODEL_NAME, device_map={"": "cuda:0"},
    torch_dtype=torch.float16, trust_remote_code=True,
)
base_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
base_model.language_model = PeftModel.from_pretrained(base_model.language_model, LORA_DIR)
base_model.eval()
print(f"✅ 微调模型加载完成  显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# ==================== 3. 评估 ====================
eval_result = evaluate_internvl2(
    base_model, tokenizer, raw_test, IMAGE_ROOT,
    system_prompt=SYSTEM_PROMPT,
    sbert_path=SBERT_PATH,
    transform=transform,
    img_context_token=IMG_CONTEXT_TOKEN,
    num_visual_tokens=NUM_VISUAL_TOKENS,
    desc="InternVL2 微调模型评估",
    eval_batch_size=EVAL_BATCH_SIZE,
)

print_results("InternVL2-4B 微调模型评估结果", eval_result)

with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(eval_result, f, indent=2, ensure_ascii=False)
print(f"\n✅ 详细结果已保存到 {EVAL_OUTPUT}")
