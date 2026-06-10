"""
通用评估工具函数
- Gemma3:      批量推理（原生支持）
- Qwen2.5-VL:  批量推理（左填充，官方方式）
- InternVL2:   批量推理（model.batch_chat）
"""

import re
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ==================== 共用工具 ====================

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


def get_rule_set(violations):
    rules = set()
    for v in violations:
        try:
            rules.add(int(v["rule"]))
        except:
            pass
    return rules


def compute_metrics(tp, fp, fn, results, parse_failures, rouge_scores, sbert_scores):
    all_rules = [1, 2, 3, 4]

    valid_results = [r for r in results if not r.get("parse_failed", False)]
    total         = len(results)
    valid_count   = len(valid_results)

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

    total_tp = sum(tp[r] for r in all_rules)
    total_fp = sum(fp[r] for r in all_rules)
    total_fn = sum(fn[r] for r in all_rules)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    exact_match  = sum(1 for r in valid_results
                       if set(r["gt_rules"]) == set(r["pred_rules"]))
    safe_correct = sum(1 for r in valid_results
                       if (len(r["gt_rules"]) == 0) == (len(r["pred_rules"]) == 0))

    exact_match_acc = exact_match  / valid_count if valid_count > 0 else 0
    safe_unsafe_acc = safe_correct / valid_count if valid_count > 0 else 0

    return {
        "summary": {
            "total_samples":      total,
            "valid_samples":      valid_count,
            "parse_failures":     parse_failures,
            "parse_failure_rate": parse_failures / total if total > 0 else 0,
            "exact_match_acc":    exact_match_acc,
            "safe_unsafe_acc":    safe_unsafe_acc,
            "macro_precision":    float(macro_p),
            "macro_recall":       float(macro_re),
            "macro_f1":           float(macro_f1),
            "micro_precision":    float(micro_p),
            "micro_recall":       float(micro_r),
            "micro_f1":           float(micro_f1),
            "avg_rouge_l":        float(np.mean(rouge_scores)) if rouge_scores else 0,
            "avg_sbert_sim":      float(np.mean(sbert_scores)) if sbert_scores else 0,
        },
        "per_rule": {str(r): per_rule[r] for r in all_rules},
        "details":  results,
    }


def print_results(label, eval_result):
    s = eval_result["summary"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"\n[样本统计]")
    print(f"  总样本:     {s['total_samples']}")
    print(f"  有效样本:   {s['valid_samples']}")
    print(f"  Parse 失败: {s['parse_failures']} ({s['parse_failure_rate']*100:.1f}%)")
    print(f"\n[Rule Violation 分类指标]")
    print(f"{'Rule':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 56)
    for r in [1, 2, 3, 4]:
        pr = eval_result["per_rule"][str(r)]
        print(f"Rule {r:<3} {pr['precision']:>10.4f} {pr['recall']:>10.4f} "
              f"{pr['f1']:>10.4f} {pr['tp']:>6} {pr['fp']:>6} {pr['fn']:>6}")
    print("-" * 56)
    print(f"{'Macro':<8} {s['macro_precision']:>10.4f} {s['macro_recall']:>10.4f} {s['macro_f1']:>10.4f}")
    print(f"{'Micro':<8} {s['micro_precision']:>10.4f} {s['micro_recall']:>10.4f} {s['micro_f1']:>10.4f}")
    print(f"\n[整体准确率]（仅有效样本）")
    print(f"  Exact Match:  {s['exact_match_acc']*100:.2f}%")
    print(f"  Safe/Unsafe:  {s['safe_unsafe_acc']*100:.2f}%")
    print(f"\n[Annotation 文本相似度]")
    print(f"  ROUGE-L:   {s['avg_rouge_l']:.4f}")
    print(f"  SBERT sim: {s['avg_sbert_sim']:.4f}")


def _process_pred(results, rouge_scores, sbert_scores, tp, fp, fn,
                  all_rules, sample, pred_text, gt_asst, rouge, sbert):
    """解析单条输出并更新指标，返回 parse_failures_delta"""
    from sentence_transformers import util as st_util

    pred_json, parse_ok = parse_json_output(pred_text)
    gt_json,   _        = parse_json_output(gt_asst)

    if not parse_ok:
        results.append({
            "image":           sample["image"],
            "gt_annotation":   gt_json.get("annotation", ""),
            "pred_annotation": "",
            "gt_rules":        list(get_rule_set(gt_json.get("violations", []))),
            "pred_rules":      [],
            "rouge_l":         0.0,
            "sbert_sim":       0.0,
            "pred_raw":        pred_text,
            "parse_failed":    True,
        })
        return 1

    pred_annotation = pred_json.get("annotation", "")
    gt_annotation   = gt_json.get("annotation", "")
    pred_rules      = get_rule_set(pred_json.get("violations", []))
    gt_rules        = get_rule_set(gt_json.get("violations", []))

    for r in all_rules:
        pred_pos = r in pred_rules
        gt_pos   = r in gt_rules
        if pred_pos and gt_pos:       tp[r] += 1
        elif pred_pos and not gt_pos: fp[r] += 1
        elif not pred_pos and gt_pos: fn[r] += 1

    rouge_l   = rouge.score(gt_annotation, pred_annotation)["rougeL"].fmeasure
    rouge_scores.append(rouge_l)
    sbert_sim = st_util.cos_sim(
        sbert.encode(pred_annotation, convert_to_tensor=True),
        sbert.encode(gt_annotation,   convert_to_tensor=True),
    ).item()
    sbert_scores.append(sbert_sim)

    results.append({
        "image":           sample["image"],
        "gt_annotation":   gt_annotation,
        "pred_annotation": pred_annotation,
        "gt_rules":        list(gt_rules),
        "pred_rules":      list(pred_rules),
        "rouge_l":         rouge_l,
        "sbert_sim":       sbert_sim,
        "pred_raw":        pred_text,
        "parse_failed":    False,
    })
    return 0


# ==================== Gemma3：批量推理 ====================

def evaluate_gemma3(model, processor, test_data, image_root,
                    system_prompt, sbert_path,
                    desc="评估中", eval_batch_size=8):
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer

    rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sbert  = SentenceTransformer(sbert_path)
    device = torch.device("cuda:0")

    all_rules      = [1, 2, 3, 4]
    tp = {r: 0 for r in all_rules}
    fp = {r: 0 for r in all_rules}
    fn = {r: 0 for r in all_rules}
    rouge_scores   = []
    sbert_scores   = []
    results        = []
    parse_failures = 0

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
        conv  = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        prompt_text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        prepared.append({"sample": sample, "image": image,
                         "prompt_text": prompt_text, "gt_asst": gt_asst})

    model.eval()
    for batch_start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        batch_items  = prepared[batch_start: batch_start + eval_batch_size]
        batch_inputs = processor(
            text   = [item["prompt_text"] for item in batch_items],
            images = [[item["image"]] for item in batch_items],
            return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **batch_inputs, max_new_tokens=1024,
                    do_sample=False, top_p=None, top_k=None,
                )

        for i, item in enumerate(batch_items):
            real_len  = batch_inputs["attention_mask"][i].sum().item()
            generated = output_ids[i][real_len:]
            pred_text = processor.decode(generated, skip_special_tokens=True).strip()
            parse_failures += _process_pred(
                results, rouge_scores, sbert_scores, tp, fp, fn,
                all_rules, item["sample"], pred_text, item["gt_asst"], rouge, sbert,
            )

    return compute_metrics(tp, fp, fn, results, parse_failures, rouge_scores, sbert_scores)


# ==================== Qwen2.5-VL：批量推理（左填充） ====================

def evaluate_qwen25vl(model, processor, test_data, image_root,
                      system_prompt, sbert_path,
                      desc="评估中", eval_batch_size=8):
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from qwen_vl_utils import process_vision_info

    rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sbert  = SentenceTransformer(sbert_path)
    device = torch.device("cuda:0")

    all_rules      = [1, 2, 3, 4]
    tp = {r: 0 for r in all_rules}
    fp = {r: 0 for r in all_rules}
    fn = {r: 0 for r in all_rules}
    rouge_scores   = []
    sbert_scores   = []
    results        = []
    parse_failures = 0

    # 关键：设置左填充
    processor.tokenizer.padding_side = "left"

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
        # 官方批量推理格式：每条是一个 messages list
        conv = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        text = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conv)
        prepared.append({
            "sample":        sample,
            "text":          text,
            "image_inputs":  image_inputs,
            "video_inputs":  video_inputs,
            "gt_asst":       gt_asst,
        })

    model.eval()
    for batch_start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        batch_items = prepared[batch_start: batch_start + eval_batch_size]

        texts        = [item["text"] for item in batch_items]
        # 合并所有图像 inputs
        all_image_inputs = []
        for item in batch_items:
            if item["image_inputs"]:
                all_image_inputs.extend(item["image_inputs"])

        batch_inputs = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output_ids = model.generate(
                    **batch_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

        # 用 input_ids 长度切掉输入部分
        input_len = batch_inputs["input_ids"].shape[1]
        for i, item in enumerate(batch_items):
            generated = output_ids[i][input_len:]
            pred_text = processor.decode(generated, skip_special_tokens=True).strip()
            parse_failures += _process_pred(
                results, rouge_scores, sbert_scores, tp, fp, fn,
                all_rules, item["sample"], pred_text, item["gt_asst"], rouge, sbert,
            )

    # 恢复默认 padding side
    processor.tokenizer.padding_side = "right"

    return compute_metrics(tp, fp, fn, results, parse_failures, rouge_scores, sbert_scores)


# ==================== InternVL2：批量推理（model.batch_chat） ====================

def evaluate_internvl2(model, tokenizer, test_data, image_root,
                       system_prompt, sbert_path,
                       transform, img_context_token, num_visual_tokens,
                       desc="评估中", eval_batch_size=8):
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer

    rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sbert  = SentenceTransformer(sbert_path)
    device = torch.device("cuda:0")

    all_rules      = [1, 2, 3, 4]
    tp = {r: 0 for r in all_rules}
    fp = {r: 0 for r in all_rules}
    fn = {r: 0 for r in all_rules}
    rouge_scores   = []
    sbert_scores   = []
    results        = []
    parse_failures = 0

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
        question     = f"{system_prompt}\n\n{user_text}"
        prepared.append({
            "sample":       sample,
            "pixel_values": pixel_values,
            "question":     question,
            "gt_asst":      gt_asst,
        })

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    model.eval()
    for batch_start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        batch_items      = prepared[batch_start: batch_start + eval_batch_size]
        pixel_values     = torch.cat([item["pixel_values"] for item in batch_items], dim=0).to(device)
        questions        = [item["question"] for item in batch_items]
        num_patches_list = [1] * len(batch_items)   # 每张图 1 组 patch

        with torch.no_grad():
            pred_texts = model.batch_chat(
                tokenizer        = tokenizer,
                pixel_values     = pixel_values,
                questions        = questions,
                num_patches_list = num_patches_list,
                generation_config = generation_config,
            )

        for i, item in enumerate(batch_items):
            pred_text = pred_texts[i]
            parse_failures += _process_pred(
                results, rouge_scores, sbert_scores, tp, fp, fn,
                all_rules, item["sample"], pred_text, item["gt_asst"], rouge, sbert,
            )

    return compute_metrics(tp, fp, fn, results, parse_failures, rouge_scores, sbert_scores)