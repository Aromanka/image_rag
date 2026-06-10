"""
独立评估脚本：加载微调后的 Gemma3-4B LoRA 模型，在 test.json 上评估
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from evaluate_utils import evaluate_gemma3, print_results

# ==================== 0. 配置 ====================
MODEL_NAME        = "/root/autodl-tmp/gemma-3-4b-it"
LORA_DIR          = "/root/autodl-tmp/gemma3_4b_lora"
TEST_JSON         = "/root/autodl-tmp/test.json"
IMAGE_ROOT        = "/root/autodl-tmp/images"
SBERT_PATH        = "/root/autodl-tmp/all-MiniLM-L6-v2"
EVAL_OUTPUT       = "/root/autodl-tmp/eval_gemma3_finetuned.json"

MAX_TEST_SAMPLES  = None
EVAL_BATCH_SIZE   = 8

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

# ==================== 1. 加载测试数据 ====================
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
print("🚀 加载微调后的 Gemma3-4B 模型...")
processor  = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()
print(f"✅ 微调模型加载完成  显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# ==================== 3. 评估 ====================
eval_result = evaluate_gemma3(
    model, processor, raw_test, IMAGE_ROOT,
    system_prompt=SYSTEM_PROMPT,
    sbert_path=SBERT_PATH,
    desc="Gemma3 微调模型评估",
    eval_batch_size=EVAL_BATCH_SIZE,
)

print_results("Gemma3-4B 微调模型评估结果", eval_result)

with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(eval_result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 详细结果已保存到 {EVAL_OUTPUT}")
