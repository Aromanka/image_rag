"""
Qwen2.5-VL-3B LoRA 微调 + 测试评估（基础模型 vs 微调模型）
适配 ConstructionSite10k 数据集
评估指标：
  1. Rule violation 分类准确率 (Precision/Recall/F1)
  2. Annotation 文本相似度 (ROUGE-L + SBERT)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import re
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from qwen_vl_utils import process_vision_info

# ==================== 0. 配置 ====================
MODEL_NAME        = "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct"
TRAIN_JSON        = "/root/autodl-tmp/train.json"
TEST_JSON         = "/root/autodl-tmp/test.json"
IMAGE_ROOT        = "/root/autodl-tmp/images"
OUTPUT_DIR        = "/root/autodl-tmp/outputs_qwen25vl"
LORA_DIR          = "/root/autodl-tmp/qwen25vl_3b_lora"
EVAL_OUTPUT       = "/root/autodl-tmp/eval_results_qwen25vl.json"
SBERT_PATH        = "/root/autodl-tmp/all-MiniLM-L6-v2"

BATCH_SIZE        = 1
GRAD_ACCUM_STEPS  = 8
NUM_EPOCHS        = 1
LEARNING_RATE     = 2e-4
MAX_LENGTH        = 2048
LORA_R            = 16
LORA_ALPHA        = 32
WARMUP_STEPS      = 50
MAX_TRAIN_SAMPLES = None
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

# ==================== 工具函数 ====================
def load_data(json_path, max_samples=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        sample["image"] = sample["image"].replace("\\", "/")
    if max_samples:
        data = data[:max_samples]
    return data

def parse_json_output(text):
    try:
        return json.loads(text.strip())
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"annotation": "", "violations": []}

def get_rule_set(violations):
    rules = set()
    for v in violations:
        try:
            rules.add(int(v["rule"]))
        except:
            pass
    return rules

def free_gpu_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)
    print(f"  显存释放后: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

def load_base_model():
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
        min_pixels=128 * 28 * 28,    # 加这两行限制分辨率
        max_pixels=256 * 28 * 28,
    )
    #processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model, processor

# ==================== 评估函数 ====================
def evaluate_model(model, processor, test_data, image_root,
                   desc="评估中", eval_batch_size=8):
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util

    rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sbert  = SentenceTransformer(SBERT_PATH)
    device = torch.device("cuda:0")

    all_rules    = [1, 2, 3, 4]
    tp = {r: 0 for r in all_rules}
    fp = {r: 0 for r in all_rules}
    fn = {r: 0 for r in all_rules}
    rouge_scores = []
    sbert_scores = []
    results      = []

    # 预处理所有样本
    prepared = []
    for sample in test_data:
        messages  = sample["messages"]
        user_text = ""
        img_path  = None
        gt_asst   = ""

        for msg in messages:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_path = Path(image_root) / Path(sample["image"]).name
                    elif item["type"] == "text":
                        user_text = item["text"].strip()
            elif msg["role"] == "assistant":
                gt_asst = msg["content"].strip()

        image = Image.open(img_path).convert("RGB")

        conv = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        text = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(conv)

        prepared.append({
            "sample":       sample,
            "image":        image,
            "image_inputs": image_inputs,
            "text":         text,
            "gt_asst":      gt_asst,
        })

    # 批量推理
    model.eval()
    for batch_start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        batch_items = prepared[batch_start: batch_start + eval_batch_size]

        texts        = [item["text"] for item in batch_items]
        all_images   = [item["image_inputs"][0] for item in batch_items]

        batch_inputs = processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output_ids = model.generate(
                    **batch_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

        input_len = batch_inputs["input_ids"].shape[1]
        for i, item in enumerate(batch_items):
            generated = output_ids[i][input_len:]
            pred_text = processor.decode(generated, skip_special_tokens=True).strip()

            pred_json = parse_json_output(pred_text)
            gt_json   = parse_json_output(item["gt_asst"])

            pred_annotation = pred_json.get("annotation", "")
            gt_annotation   = gt_json.get("annotation", "")
            pred_rules      = get_rule_set(pred_json.get("violations", []))
            gt_rules        = get_rule_set(gt_json.get("violations", []))

            for r in all_rules:
                pred_pos = r in pred_rules
                gt_pos   = r in gt_rules
                if pred_pos and gt_pos:
                    tp[r] += 1
                elif pred_pos and not gt_pos:
                    fp[r] += 1
                elif not pred_pos and gt_pos:
                    fn[r] += 1

            rouge_l   = rouge.score(gt_annotation, pred_annotation)["rougeL"].fmeasure
            rouge_scores.append(rouge_l)

            emb_pred  = sbert.encode(pred_annotation, convert_to_tensor=True)
            emb_gt    = sbert.encode(gt_annotation,   convert_to_tensor=True)
            sbert_sim = util.cos_sim(emb_pred, emb_gt).item()
            sbert_scores.append(sbert_sim)

            results.append({
                "image":           item["sample"]["image"],
                "gt_annotation":   gt_annotation,
                "pred_annotation": pred_annotation,
                "gt_rules":        list(gt_rules),
                "pred_rules":      list(pred_rules),
                "rouge_l":         rouge_l,
                "sbert_sim":       sbert_sim,
                "pred_raw":        pred_text,
            })

    exact_match  = sum(1 for r in results if set(r["gt_rules"]) == set(r["pred_rules"]))
    safe_correct = sum(1 for r in results if (len(r["gt_rules"])==0) == (len(r["pred_rules"])==0))

    per_rule = {}
    for r in all_rules:
        p  = tp[r] / (tp[r] + fp[r]) if (tp[r] + fp[r]) > 0 else 0
        rc = tp[r] / (tp[r] + fn[r]) if (tp[r] + fn[r]) > 0 else 0
        f1 = 2 * p * rc / (p + rc)   if (p + rc) > 0 else 0
        per_rule[r] = {"tp": tp[r], "fp": fp[r], "fn": fn[r],
                       "precision": p, "recall": rc, "f1": f1}

    macro_p  = np.mean([per_rule[r]["precision"] for r in all_rules])
    macro_re = np.mean([per_rule[r]["recall"]    for r in all_rules])
    macro_f1 = np.mean([per_rule[r]["f1"]        for r in all_rules])

    return {
        "summary": {
            "exact_match_acc": exact_match / len(results),
            "safe_unsafe_acc": safe_correct / len(results),
            "macro_precision": float(macro_p),
            "macro_recall":    float(macro_re),
            "macro_f1":        float(macro_f1),
            "avg_rouge_l":     float(np.mean(rouge_scores)),
            "avg_sbert_sim":   float(np.mean(sbert_scores)),
        },
        "per_rule": {str(r): per_rule[r] for r in all_rules},
        "details":  results,
    }

def print_results(label, eval_result):
    s = eval_result["summary"]
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"\n[Rule Violation 分类指标]")
    print(f"{'Rule':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for r in [1, 2, 3, 4]:
        pr = eval_result["per_rule"][str(r)]
        print(f"Rule {r:<3} {pr['precision']:>10.4f} {pr['recall']:>10.4f} {pr['f1']:>10.4f}")
    print("-" * 42)
    print(f"{'Macro':<8} {s['macro_precision']:>10.4f} {s['macro_recall']:>10.4f} {s['macro_f1']:>10.4f}")
    print(f"\nExact Match 准确率:    {s['exact_match_acc']*100:.2f}%")
    print(f"Safe/Unsafe 准确率:    {s['safe_unsafe_acc']*100:.2f}%")
    print(f"\n[Annotation 文本相似度]")
    print(f"  ROUGE-L:   {s['avg_rouge_l']:.4f}")
    print(f"  SBERT sim: {s['avg_sbert_sim']:.4f}")


# ==================== 安装评估依赖 ====================
os.system("pip install rouge-score sentence-transformers -q")

# ==================== 加载数据 ====================
raw_train = load_data(TRAIN_JSON, MAX_TRAIN_SAMPLES)
raw_test  = load_data(TEST_JSON,  MAX_TEST_SAMPLES)
print(f"✅ 训练集: {len(raw_train)} 条  测试集: {len(raw_test)} 条")


# ============================================================
# STEP 1: 测试基础模型（微调前）
# ============================================================
print("\n" + "="*55)
print("STEP 1: 加载基础模型并评估")
print("="*55)

base_model, processor = load_base_model()
print(f"基础模型加载后显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

#base_results = evaluate_model(
#    base_model, processor, raw_test, IMAGE_ROOT,
#    desc="基础模型评估",
#    eval_batch_size=EVAL_BATCH_SIZE,
#)
#print_results("基础模型（微调前）", base_results)

from evaluate_utils import evaluate_qwen25vl, print_results
base_results = evaluate_qwen25vl(
    base_model, processor, raw_test, IMAGE_ROOT,
    system_prompt=SYSTEM_PROMPT, sbert_path=SBERT_PATH,
    desc="基础模型评估", eval_batch_size=EVAL_BATCH_SIZE,
)

print_results("基础模型（微调前）", base_results)


print("\n释放基础模型显存...")
free_gpu_memory(base_model)


# ============================================================
# STEP 2: LoRA 微调
# ============================================================
print("\n" + "="*55)
print("STEP 2: LoRA 微调")
print("="*55)

model, processor = load_base_model()

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad_(True)
model.print_trainable_parameters()


class ConstructionDataset(Dataset):
    def __init__(self, data, processor, image_root, max_length):
        self.data       = data
        self.processor  = processor
        self.image_root = Path(image_root)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample   = self.data[idx]
        messages = sample["messages"]

        qwen_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # 替换成优化后的 system prompt
                qwen_messages.append({"role": "system", "content": SYSTEM_PROMPT})
            elif msg["role"] == "user":
                content = []
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_path = self.image_root / Path(sample["image"]).name
                        image    = Image.open(img_path).convert("RGB")
                        image    = image.resize((448, 448))
                        content.append({"type": "image", "image": image})
                    else:
                        content.append(item)
                qwen_messages.append({"role": "user", "content": content})
            elif msg["role"] == "assistant":
                qwen_messages.append({"role": "assistant", "content": msg["content"]})

        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        assistant_tokens = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        seq = labels.tolist()
        n   = len(assistant_tokens)
        mask_end = 0
        for i in range(len(seq) - n):
            if seq[i: i + n] == assistant_tokens:
                mask_end = i + n
                break
        labels[:mask_end] = -100
        inputs["labels"] = labels
        return inputs


train_dataset = ConstructionDataset(
    data=raw_train, processor=processor,
    image_root=IMAGE_ROOT, max_length=MAX_LENGTH,
)

def collate_fn(features):
    keys = features[0].keys()
    batch = {}
    for key in keys:
        tensors = [f[key] for f in features]
        if key == "labels":
            max_len = max(t.size(0) for t in tensors)
            padded  = torch.full((len(tensors), max_len), -100, dtype=torch.long)
            for i, t in enumerate(tensors):
                padded[i, :t.size(0)] = t
            batch[key] = padded
        elif tensors[0].dtype in (torch.long, torch.int):
            max_len = max(t.size(0) for t in tensors)
            padded  = torch.zeros(len(tensors), max_len, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, :t.size(0)] = t
            batch[key] = padded
        else:
            try:
                batch[key] = torch.stack(tensors)
            except Exception:
                batch[key] = tensors
    return batch

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=4, pin_memory=True,
)

optimizer   = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=0.01,
)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
scheduler   = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS,
    num_training_steps=max(total_steps, 1),
)

print(f"\n🏋️ 开始训练  epochs={NUM_EPOCHS}  total_steps={total_steps}")
print(f"GPU 总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device      = torch.device("cuda:0")
scaler      = torch.cuda.amp.GradScaler()
global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_loss = epoch_steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        if step == 0 and epoch == 0:
            print("pixel_values shape:", batch["pixel_values"].shape)
            print("image_grid_thw:", batch.get("image_grid_thw"))
            img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            print("image tokens in batch:", (batch["input_ids"] == img_token_id).sum().item())

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**batch)
            loss    = outputs.loss / GRAD_ACCUM_STEPS
            
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Step {step} Epoch {epoch+1} loss=nan，跳过此 batch")
            optimizer.zero_grad()
            scaler.update()
            continue

        scaler.scale(loss).backward()
        epoch_loss  += loss.item() * GRAD_ACCUM_STEPS
        epoch_steps += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.set_postfix({
                "loss": f"{epoch_loss/epoch_steps:.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                "mem":  f"{torch.cuda.memory_reserved()/1024**3:.1f}GB",
            })

    if epoch_steps % GRAD_ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print(f"\n📊 Epoch {epoch+1} 完成  avg_loss={epoch_loss/epoch_steps:.4f}")
    ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    print(f"💾 已保存到 {ckpt_dir}")
    gc.collect()
    torch.cuda.empty_cache()

model.save_pretrained(LORA_DIR)
processor.save_pretrained(LORA_DIR)
print(f"\n🎉 训练完成！LoRA 已保存到 {LORA_DIR}")
print(f"训练峰值显存: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")


