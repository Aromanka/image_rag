"""
InternVL2-4B LoRA 微调 + 测试评估（基础模型 vs 微调模型）
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
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ==================== 0. 配置 ====================
MODEL_NAME        = "/root/autodl-tmp/InternVL2-4B"
TRAIN_JSON        = "/root/autodl-tmp/train.json"
TEST_JSON         = "/root/autodl-tmp/test.json"
IMAGE_ROOT        = "/root/autodl-tmp/images"
OUTPUT_DIR        = "/root/autodl-tmp/outputs_internvl2"
LORA_DIR          = "/root/autodl-tmp/internvl2_4b_lora"
EVAL_OUTPUT       = "/root/autodl-tmp/eval_results_internvl2.json"
SBERT_PATH        = "/root/autodl-tmp/all-MiniLM-L6-v2"

BATCH_SIZE        = 1
GRAD_ACCUM_STEPS  = 8
NUM_EPOCHS        = 1
LEARNING_RATE     = 2e-4
MAX_LENGTH        = 2048
IMG_SIZE          = 448
LORA_R            = 16
LORA_ALPHA        = 32
WARMUP_STEPS      = 50
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES  = None
EVAL_BATCH_SIZE   = 16

IMAGENET_MEAN     = (0.485, 0.456, 0.406)
IMAGENET_STD      = (0.229, 0.224, 0.225)
NUM_VISUAL_TOKENS = 256
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

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
def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

transform = build_transform()

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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model, tokenizer

def build_internvl_prompt(system_text, user_text):
    """构建 InternVL2 推理用的 prompt（不含图像 placeholder，用 chat() 方法时自动处理）"""
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text

# ==================== 评估函数 ====================
def evaluate_model(model, tokenizer, test_data, image_root,
                   desc="评估中", eval_batch_size=18):
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

        image        = Image.open(img_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(torch.float16)

        img_placeholder = "<img>" + IMG_CONTEXT_TOKEN * NUM_VISUAL_TOKENS + "</img>"
        prompt = (f"<|system|>\n{SYSTEM_PROMPT}</s>"
                  f"<|user|>\n{img_placeholder}\n{user_text}</s>"
                  f"<|assistant|>\n")

        prepared.append({
            "sample":       sample,
            "pixel_values": pixel_values,
            "prompt":       prompt,
            "gt_asst":      gt_asst,
        })

    # 批量推理
    model.eval()
    for batch_start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        batch_items = prepared[batch_start: batch_start + eval_batch_size]

        pixel_values = torch.cat(
            [item["pixel_values"] for item in batch_items], dim=0
        ).to(device)

        # image_flags: 每张图 1 个 flag，shape [B, 1]
        image_flags = torch.ones(
            len(batch_items), 1, dtype=torch.long
        ).to(device)

        # 逐条 tokenize 再右对齐 padding
        all_enc = []
        for item in batch_items:
            enc = tokenizer(
                item["prompt"],
                return_tensors="pt",
                add_special_tokens=False,
            )
            all_enc.append(enc)

        max_len        = max(e["input_ids"].shape[1] for e in all_enc)
        input_ids      = torch.zeros(len(batch_items), max_len, dtype=torch.long).to(device)
        attention_mask = torch.zeros(len(batch_items), max_len, dtype=torch.long).to(device)

        for i, enc in enumerate(all_enc):
            seq_len = enc["input_ids"].shape[1]
            input_ids[i, -seq_len:]      = enc["input_ids"][0]
            attention_mask[i, -seq_len:] = enc["attention_mask"][0]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output_ids = model.generate(
                    pixel_values   = pixel_values,
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = 512,
                    do_sample      = False,
                )

        # 解码每条输出
        for i, item in enumerate(batch_items):

            real_input_len = attention_mask[i].sum().item()
            generated  = output_ids[i]
            pred_text  = tokenizer.decode(generated, skip_special_tokens=True).strip()

            pred_json = parse_json_output(pred_text)
            gt_json   = parse_json_output(item["gt_asst"])

            pred_annotation = pred_json.get("annotation", "")
            gt_annotation   = gt_json.get("annotation", "")
            pred_rules      = get_rule_set(pred_json.get("violations", []))
            gt_rules        = get_rule_set(gt_json.get("violations", []))
            if batch_start == 0 and i < 3:
                print(f"output_ids[i] length: {len(output_ids[i])}")
                print(f"real_input_len: {real_input_len}")
                print(f"attention_mask[i].sum(): {attention_mask[i].sum().item()}")
                print(f"input_ids shape: {input_ids.shape}")

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

    # 汇总指标
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

base_model, tokenizer = load_base_model()
print(f"基础模型加载后显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# 在训练开始前加这段调试代码
sample = raw_train[0]
img_path = Path(IMAGE_ROOT) / Path(sample["image"]).name
image = Image.open(img_path).convert("RGB")

img_placeholder = "<img>" + IMG_CONTEXT_TOKEN * NUM_VISUAL_TOKENS + "</img>"
prompt = f"<|system|>\n{SYSTEM_PROMPT}</s><|user|>\n{img_placeholder}\ntest</s><|assistant|>\n"

enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
input_ids = enc["input_ids"].squeeze(0)

img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
print("IMG_CONTEXT_TOKEN:", IMG_CONTEXT_TOKEN)
print("img_context_id:", img_context_id)
print("matches in sequence:", (input_ids == img_context_id).sum().item())
print("expected:", NUM_VISUAL_TOKENS)

# 在基础模型评估之前加这段，只看前3条的原始输出
base_model.eval()
for i, sample in enumerate(raw_test[:3]):
    for msg in sample["messages"]:
        if msg["role"] == "user":
            for item in msg["content"]:
                if item["type"] == "image":
                    img_path = Path(IMAGE_ROOT) / Path(sample["image"]).name
                elif item["type"] == "text":
                    user_text = item["text"].strip()

    image = Image.open(img_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(torch.float16).to("cuda:0")

    with torch.no_grad():
        pred = base_model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=f"{SYSTEM_PROMPT}\n\n{user_text}",
            generation_config=dict(max_new_tokens=1024, do_sample=False),
        )
    print(f"\n=== Sample {i} ===")
    print(pred[:500])



#base_results = evaluate_model(
#    base_model, tokenizer, raw_test, IMAGE_ROOT,
#    desc="基础模型评估",
#)
#print_results("基础模型（微调前）", base_results)
from evaluate_utils import evaluate_internvl2, print_results
base_results = evaluate_internvl2(
    base_model, tokenizer, raw_test, IMAGE_ROOT,
    system_prompt=SYSTEM_PROMPT, sbert_path=SBERT_PATH,
    transform=transform, img_context_token=IMG_CONTEXT_TOKEN,
    num_visual_tokens=NUM_VISUAL_TOKENS,
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

model, tokenizer = load_base_model()

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model.language_model = get_peft_model(model.language_model, lora_config)
model.language_model.enable_input_require_grads()

for name, param in model.named_parameters():
    if "vision_model" in name or "mlp1" in name:
        param.requires_grad_(False)
    if "lora" in name.lower():
        param.requires_grad_(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")


class ConstructionDataset(Dataset):
    def __init__(self, data, tokenizer, image_root, max_length):
        self.data           = data
        self.tokenizer      = tokenizer
        self.image_root     = Path(image_root)
        self.max_length     = max_length
        self.img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample   = self.data[idx]
        messages = sample["messages"]
        system_text = user_text = asst_text = ""
        img_path = None

        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"].strip()
            elif msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_path = self.image_root / Path(sample["image"]).name
                    elif item["type"] == "text":
                        user_text = item["text"].strip()
            elif msg["role"] == "assistant":
                asst_text = msg["content"].strip()

        image        = Image.open(img_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(torch.float16)

        img_placeholder = "<img>" + IMG_CONTEXT_TOKEN * NUM_VISUAL_TOKENS + "</img>"
        system_to_use   = SYSTEM_PROMPT if not system_text else system_text
        prompt = (f"<|system|>\n{system_to_use}</s>"
                  f"<|user|>\n{img_placeholder}\n{user_text}</s>"
                  f"<|assistant|>\n")
        full   = prompt + asst_text + "</s>"

        enc = self.tokenizer(
            full, return_tensors="pt", truncation=True,
            max_length=self.max_length, add_special_tokens=False,
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels     = input_ids.clone()
        labels[:prompt_len] = -100

        image_flags = torch.ones(1, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "pixel_values":   pixel_values.squeeze(0),
            "image_flags":    image_flags,
        }


train_dataset = ConstructionDataset(
    data=raw_train, tokenizer=tokenizer,
    image_root=IMAGE_ROOT, max_length=MAX_LENGTH,
)

def collate_fn(features):
    max_len = max(f["input_ids"].size(0) for f in features)
    B = len(features)
    batch_input_ids      = torch.zeros(B, max_len, dtype=torch.long)
    batch_attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    batch_labels         = torch.full((B, max_len), -100, dtype=torch.long)
    image_flags_list = [f["image_flags"] for f in features]
    batch_image_flags = torch.stack(image_flags_list)
    pixel_list = []
    for i, f in enumerate(features):
        n = f["input_ids"].size(0)
        batch_input_ids[i, :n]      = f["input_ids"]
        batch_attention_mask[i, :n] = f["attention_mask"]
        batch_labels[i, :n]         = f["labels"]
        batch_image_flags[i, :n]    = f["image_flags"]
        pixel_list.append(f["pixel_values"])
    return {
    "input_ids":      batch_input_ids,
    "attention_mask": batch_attention_mask,
    "labels":         batch_labels,
    "pixel_values":   torch.stack(pixel_list),
    "image_flags":    batch_image_flags,   # [B, 1]
    }

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
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(
                pixel_values   = batch["pixel_values"],
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                image_flags    = batch["image_flags"],
                labels         = batch["labels"],
            )
            if step == 0 and epoch == 0:
                print("pixel_values shape:", batch["pixel_values"].shape)
                img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                selected_count = (batch["input_ids"] == img_ctx_id).sum().item()
                print("IMG_CONTEXT tokens in batch:", selected_count)
                print("expected:", batch["pixel_values"].shape[0] * 256)
                print("labels non -100 count:", (batch["labels"] != -100).sum().item())
            loss = outputs.loss / GRAD_ACCUM_STEPS

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
    model.language_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"💾 已保存到 {ckpt_dir}")
    gc.collect()
    torch.cuda.empty_cache()

model.language_model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"\n🎉 训练完成！LoRA 已保存到 {LORA_DIR}")
print(f"训练峰值显存: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")


# ============================================================
# STEP 3: 重新加载微调模型并评估
# ============================================================
print("\n" + "="*55)
print("STEP 3: 重新加载微调模型并评估")
print("="*55)

print("释放训练显存...")
free_gpu_memory(model, optimizer, scheduler)

print("重新加载微调模型...")
ft_base, tokenizer = load_base_model()
ft_base.language_model = PeftModel.from_pretrained(ft_base.language_model, LORA_DIR)
ft_base.eval()
print(f"微调模型加载后显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

ft_results = evaluate_model(
    ft_base, tokenizer, raw_test, IMAGE_ROOT,
    desc="微调模型评估",
)
print_results("微调后模型", ft_results)

free_gpu_memory(ft_base)


# ============================================================
# STEP 4: 对比汇总
# ============================================================
print("\n\n" + "="*55)
print("  📊 基础模型 vs 微调模型 对比")
print("="*55)

metrics = ["exact_match_acc", "safe_unsafe_acc", "macro_precision",
           "macro_recall", "macro_f1", "avg_rouge_l", "avg_sbert_sim"]
labels  = ["Exact Match", "Safe/Unsafe Acc", "Macro Precision",
           "Macro Recall", "Macro F1", "ROUGE-L", "SBERT Sim"]

print(f"\n{'指标':<20} {'基础模型':>12} {'微调模型':>12} {'提升':>10}")
print("-" * 56)
for m, l in zip(metrics, labels):
    base_val = base_results["summary"][m]
    ft_val   = ft_results["summary"][m]
    delta    = ft_val - base_val
    arrow    = "↑" if delta > 0 else "↓" if delta < 0 else "-"
    print(f"{l:<20} {base_val:>12.4f} {ft_val:>12.4f} {delta:>+9.4f}{arrow}")

with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump({
        "base_model":      base_results,
        "finetuned_model": ft_results,
    }, f, indent=2, ensure_ascii=False)

print(f"\n✅ 完整结果已保存到 {EVAL_OUTPUT}")
