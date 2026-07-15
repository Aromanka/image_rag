"""
inspecsafe_eval_utils.py

Shared evaluation helpers for InspecSafe-V1 (oil_gas) fine-tuned models.
Assistant labels are JSON strings:
    {"scene_description": "...", "hazards": [...], "overall_safety_level": "Level X"}

All three evaluate_* functions use true batched inference (eval_batch_size respected).

Provides:
  load_data / oversample_anomalies / assistant_label
  extract_json / norm_level / hazard_set
  evaluate_qwen25vl / evaluate_gemma3 / evaluate_internvl2
  print_results
"""

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

LEVELS   = ["Level I", "Level II", "Level III", "Level IV"]
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data(path, max_samples=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        s["image"] = s["image"].replace("\\", "/")
    return data[:max_samples] if max_samples else data


def assistant_label(sample):
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            try:
                return json.loads(msg["content"])
            except Exception:
                return None
    return None


def oversample_anomalies(data, oversample_ratio=3):
    """Repeat anomaly instances (non-empty hazards OR level != IV) to balance classes."""
    import random
    normal, anomaly = [], []
    for sample in data:
        label = assistant_label(sample)
        is_anomaly = False
        if label:
            is_anomaly = bool(label.get("hazards")) or \
                         (label.get("overall_safety_level", "Level IV") != "Level IV")
        (anomaly if is_anomaly else normal).append(sample)

    print(f"Original train: normal={len(normal)}, anomaly={len(anomaly)}")
    oversampled = anomaly * oversample_ratio
    result = normal + oversampled
    random.shuffle(result)
    print(f"After oversample: total={len(result)}, anomaly×{oversample_ratio}={len(oversampled)}")
    return result


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------
def extract_json(text):
    """Extract the FIRST complete {...} block from model output and parse it."""
    if not text:
        return None
    # find the first '{' then walk forward counting braces to find its matching '}'
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    repaired = blob.replace("'", '"')
                    repaired = re.sub(r",\s*}", "}", repaired)
                    repaired = re.sub(r",\s*]", "]", repaired)
                    try:
                        return json.loads(repaired)
                    except Exception:
                        return None
    return None


def norm_level(s):
    if not s:
        return None
    m = re.search(r"(IV|III|II|I|[1-4]|one|two|three|four)", str(s), re.IGNORECASE)
    if not m:
        return None
    tok = m.group(1).lower()
    table = {"i": "Level I", "1": "Level I", "one": "Level I",
             "ii": "Level II", "2": "Level II", "two": "Level II",
             "iii": "Level III", "3": "Level III", "three": "Level III",
             "iv": "Level IV", "4": "Level IV", "four": "Level IV"}
    return table.get(tok)


def hazard_set(label):
    if not label:
        return set()
    hz = label.get("hazards", [])
    if isinstance(hz, list):
        return {str(h).strip().lower() for h in hz if str(h).strip()}
    return set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_sbert(sbert_path, device):
    if sbert_path and os.path.isdir(sbert_path):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(sbert_path, device=str(device))
        except Exception as e:
            print(f"  (SBERT unavailable: {e})")
    return None


def _accumulate(records, sample, gt_label, out_text,
                n_parse_ok, level_correct, tp, fp, fn,
                gt_descs, pred_descs, lvl_tp, lvl_fp, lvl_fn):
    pred = extract_json(out_text)
    if pred is not None:
        n_parse_ok += 1

    gt_lvl   = norm_level(gt_label.get("overall_safety_level"))
    pred_lvl = norm_level(pred.get("overall_safety_level")) if pred else None
    if gt_lvl is not None and gt_lvl == pred_lvl:
        level_correct += 1

    # per-level one-vs-rest TP/FP/FN for classification P/R/F1
    for lv in LEVELS:
        gt_pos   = (gt_lvl   == lv)
        pred_pos = (pred_lvl == lv)
        if gt_pos and pred_pos:
            lvl_tp[lv] += 1
        elif pred_pos and not gt_pos:
            lvl_fp[lv] += 1
        elif gt_pos and not pred_pos:
            lvl_fn[lv] += 1

    # hazard micro TP/FP/FN
    gt_hz, pred_hz = hazard_set(gt_label), hazard_set(pred)
    tp += len(gt_hz & pred_hz)
    fp += len(pred_hz - gt_hz)
    fn += len(gt_hz - pred_hz)

    gt_descs.append(str(gt_label.get("scene_description", "")))
    pred_descs.append(str(pred.get("scene_description", "")) if pred else "")

    records.append({
        "image":      sample["image"],
        "gt":         gt_label,
        "pred":       pred if pred is not None else {"_raw": out_text},
        "raw_output": out_text,
        "gt_level":   gt_lvl,
        "pred_level": pred_lvl,
    })
    return n_parse_ok, level_correct, tp, fp, fn


def _report_parse_failures(records, image_root, parse_fail_dir, desc):
    """Print parse-failed samples and optionally save their images into a zip."""
    failed = [r for r in records if r["pred"].get("_raw") is not None]
    print(f"\n  [Parse failures: {len(failed)}]  (desc={desc!r})")
    if not failed:
        return
    for i, r in enumerate(failed):
        img_name = Path(r["image"]).name
        raw      = r["pred"]["_raw"]
        print(f"\n  -- failure {i+1}/{len(failed)} --")
        print(f"  image    : {img_name}")
        print(f"  gt_level : {r['gt_level']}")
        print(f"  raw_output (first 300 chars): {raw[:300]!r}")
    if parse_fail_dir:
        import zipfile
        zip_path = Path(parse_fail_dir)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for r in failed:
                img_name = Path(r["image"]).name
                src      = Path(image_root) / img_name
                if src.exists():
                    zf.write(str(src), arcname=img_name)
        print(f"\n  images saved -> {zip_path}")


def _finalise(records, n, n_parse_ok, level_correct, tp, fp, fn,
              gt_descs, pred_descs, sbert, device, lvl_tp, lvl_fp, lvl_fn,
              image_root=None, parse_fail_dir=None, desc=""):

    # ── hazard P/R/F1 (micro over all hazard tokens) ─────────────────────
    haz_p  = tp / (tp + fp) if (tp + fp) else 0.0
    haz_r  = tp / (tp + fn) if (tp + fn) else 0.0
    haz_f1 = 2 * haz_p * haz_r / (haz_p + haz_r) if (haz_p + haz_r) else 0.0

    # ── per-level classification P/R/F1 (one-vs-rest) ────────────────────
    per_level = {}
    for lv in LEVELS:
        ltp, lfp, lfn = lvl_tp[lv], lvl_fp[lv], lvl_fn[lv]
        p  = ltp / (ltp + lfp) if (ltp + lfp) else 0.0
        r  = ltp / (ltp + lfn) if (ltp + lfn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        # accuracy for this level
        idx     = [i for i, rec in enumerate(records) if rec["gt_level"] == lv]
        correct = sum(1 for i in idx
                      if records[i]["gt_level"] == records[i]["pred_level"])
        per_level[lv] = {
            "n":         len(idx),
            "acc":       correct / len(idx) if idx else 0.0,
            "tp":        ltp, "fp": lfp, "fn": lfn,
            "precision": p, "recall": r, "f1": f1,
        }

    # ── macro: average P/R/F1 over levels that appear in ground truth ────
    active = [lv for lv in LEVELS if per_level[lv]["n"] > 0]
    macro_p  = float(np.mean([per_level[lv]["precision"] for lv in active])) if active else 0.0
    macro_r  = float(np.mean([per_level[lv]["recall"]    for lv in active])) if active else 0.0
    macro_f1 = float(np.mean([per_level[lv]["f1"]        for lv in active])) if active else 0.0

    # ── micro: aggregate TP/FP/FN across all levels ───────────────────────
    total_ltp = sum(lvl_tp[lv] for lv in LEVELS)
    total_lfp = sum(lvl_fp[lv] for lv in LEVELS)
    total_lfn = sum(lvl_fn[lv] for lv in LEVELS)
    micro_p  = total_ltp / (total_ltp + total_lfp) if (total_ltp + total_lfp) else 0.0
    micro_r  = total_ltp / (total_ltp + total_lfn) if (total_ltp + total_lfn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    # ── SBERT ─────────────────────────────────────────────────────────────
    sbert_sim = None
    if sbert is not None and n > 0:
        emb_gt   = sbert.encode(gt_descs, convert_to_tensor=True, show_progress_bar=False)
        emb_pred = sbert.encode(pred_descs, convert_to_tensor=True, show_progress_bar=False)
        cos      = torch.nn.functional.cosine_similarity(emb_gt, emb_pred).cpu().numpy()
        sbert_sim = float(np.mean(cos))

    # ── report parse failures ─────────────────────────────────────────────
    if image_root is not None:
        _report_parse_failures(records, image_root, parse_fail_dir, desc)

    return {
        "n_samples":        n,
        "json_parse_rate":  n_parse_ok / n if n else 0.0,
        "level_accuracy":   level_correct / n if n else 0.0,
        "per_level":        per_level,
        "level_macro_p":    macro_p,
        "level_macro_r":    macro_r,
        "level_macro_f1":   macro_f1,
        "level_micro_p":    micro_p,
        "level_micro_r":    micro_r,
        "level_micro_f1":   micro_f1,
        "hazard_precision": haz_p,
        "hazard_recall":    haz_r,
        "hazard_f1":        haz_f1,
        "scene_sbert_sim":  sbert_sim,
        "records":          records,
    }


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
def print_results(title, r):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)
    print(f"  samples          : {r['n_samples']}")
    print(f"  JSON parse rate  : {r['json_parse_rate']*100:.1f}%")
    print(f"  level accuracy   : {r['level_accuracy']*100:.1f}%")

    print(f"\n  [Safety Level Classification]")
    print(f"  {'Level':<10} {'n':>5} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*47}")
    for lv in LEVELS:
        if lv not in r.get("per_level", {}):
            continue
        d = r["per_level"][lv]
        print(f"  {lv:<10} {d['n']:>5} "
              f"{d['acc']*100:>6.1f}% "
              f"{d['precision']*100:>6.1f}% "
              f"{d['recall']*100:>6.1f}% "
              f"{d['f1']*100:>6.1f}%")
    print(f"  {'-'*47}")
    print(f"  {'Macro':<10} {'':>5} "
          f"{'':>7} "
          f"{r['level_macro_p']*100:>6.1f}% "
          f"{r['level_macro_r']*100:>6.1f}% "
          f"{r['level_macro_f1']*100:>6.1f}%")
    print(f"  {'Micro':<10} {'':>5} "
          f"{'':>7} "
          f"{r['level_micro_p']*100:>6.1f}% "
          f"{r['level_micro_r']*100:>6.1f}% "
          f"{r['level_micro_f1']*100:>6.1f}%")

    print(f"\n  [Hazard Detection]")
    print(f"  precision  : {r['hazard_precision']*100:.1f}%")
    print(f"  recall     : {r['hazard_recall']*100:.1f}%")
    print(f"  F1         : {r['hazard_f1']*100:.1f}%")

    if r.get("scene_sbert_sim") is not None:
        print(f"\n  [Scene Description]")
        print(f"  SBERT sim  : {r['scene_sbert_sim']:.4f}")


# ---------------------------------------------------------------------------
# Qwen2.5-VL  —  batched
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_qwen25vl(model, processor, test_data, image_root,
                      system_prompt, sbert_path=None, desc="eval",
                      eval_batch_size=8, max_new_tokens=384,
                      parse_fail_dir=None, img_size=None):
    from qwen_vl_utils import process_vision_info
    model.eval()
    processor.tokenizer.padding_side = "left"
    device     = next(model.parameters()).device
    image_root = Path(image_root)
    sbert      = _load_sbert(sbert_path, device)

    # ── prepare all samples ──────────────────────────────────────────────
    prepared = []
    for sample in test_data:
        gt_label  = assistant_label(sample) or {}
        img_path  = image_root / Path(sample["image"]).name
        image     = Image.open(img_path).convert("RGB")
        if img_size is not None:
            image = image.resize((img_size, img_size), Image.BILINEAR)
        user_text = next(
            (item["text"] for msg in sample["messages"] if msg["role"] == "user"
             for item in msg["content"] if item.get("type") == "text"), "")

        conv = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        text         = processor.apply_chat_template(conv, tokenize=False,
                                                     add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conv)
        prepared.append({"sample": sample, "gt_label": gt_label,
                          "text": text, "image_inputs": image_inputs})

    # ── batched inference ────────────────────────────────────────────────
    records = []
    n_parse_ok = level_correct = 0
    tp = fp = fn = 0
    gt_descs, pred_descs = [], []
    lvl_tp = {lv: 0 for lv in LEVELS}
    lvl_fp = {lv: 0 for lv in LEVELS}
    lvl_fn = {lv: 0 for lv in LEVELS}

    for start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        items = prepared[start: start + eval_batch_size]

        inputs = processor(
            text   = [it["text"] for it in items],
            images = [it["image_inputs"] for it in items],
            return_tensors="pt", padding=True,
        ).to(device)

        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        for i, item in enumerate(items):
            real_len  = inputs["attention_mask"][i].sum().item()
            out_ids   = gen[i][real_len:]
            out_text  = processor.decode(out_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).strip()
            n_parse_ok, level_correct, tp, fp, fn = _accumulate(
                records, item["sample"], item["gt_label"], out_text,
                n_parse_ok, level_correct, tp, fp, fn, gt_descs, pred_descs,
                lvl_tp, lvl_fp, lvl_fn)

    return _finalise(records, len(test_data), n_parse_ok, level_correct,
                     tp, fp, fn, gt_descs, pred_descs, sbert, device,
                     lvl_tp, lvl_fp, lvl_fn,
                     image_root=image_root, parse_fail_dir=parse_fail_dir, desc=desc)


# ---------------------------------------------------------------------------
# Gemma3  —  batched
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_gemma3(model, processor, test_data, image_root,
                    system_prompt, sbert_path=None, desc="eval",
                    eval_batch_size=8, max_new_tokens=384,
                    parse_fail_dir=None):
    model.eval()
    processor.tokenizer.padding_side = "left"
    device     = next(model.parameters()).device
    image_root = Path(image_root)
    sbert      = _load_sbert(sbert_path, device)

    # ── prepare ──────────────────────────────────────────────────────────
    prepared = []
    for sample in test_data:
        gt_label  = assistant_label(sample) or {}
        img_path  = image_root / Path(sample["image"]).name
        image     = Image.open(img_path).convert("RGB")
        user_text = next(
            (item["text"] for msg in sample["messages"] if msg["role"] == "user"
             for item in msg["content"] if item.get("type") == "text"), "")

        conv = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_text},
            ]},
        ]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True,
                                               tokenize=False)
        prepared.append({"sample": sample, "gt_label": gt_label,
                          "prompt": prompt, "image": image})

    # ── batched inference ────────────────────────────────────────────────
    records = []
    n_parse_ok = level_correct = 0
    tp = fp = fn = 0
    gt_descs, pred_descs = [], []
    lvl_tp = {lv: 0 for lv in LEVELS}
    lvl_fp = {lv: 0 for lv in LEVELS}
    lvl_fn = {lv: 0 for lv in LEVELS}

    for start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        items = prepared[start: start + eval_batch_size]

        inputs = processor(
            text   = [it["prompt"] for it in items],
            images = [[it["image"]] for it in items],
            return_tensors="pt", padding=True,
        ).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, top_p=None, top_k=None)

        for i, item in enumerate(items):
            real_len  = inputs["attention_mask"][i].sum().item()
            out_ids   = gen[i][real_len:]
            out_text  = processor.decode(out_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).strip()
            n_parse_ok, level_correct, tp, fp, fn = _accumulate(
                records, item["sample"], item["gt_label"], out_text,
                n_parse_ok, level_correct, tp, fp, fn, gt_descs, pred_descs,
                lvl_tp, lvl_fp, lvl_fn)

    return _finalise(records, len(test_data), n_parse_ok, level_correct,
                     tp, fp, fn, gt_descs, pred_descs, sbert, device,
                     lvl_tp, lvl_fp, lvl_fn,
                     image_root=image_root, parse_fail_dir=parse_fail_dir, desc=desc)


# ---------------------------------------------------------------------------
# InternVL2  —  batched  (manual pad + generate, no model.chat)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_internvl2(model, tokenizer, test_data, image_root,
                       system_prompt, transform,
                       img_context_token="<IMG_CONTEXT>", num_visual_tokens=256,
                       sbert_path=None, desc="eval",
                       eval_batch_size=8, max_new_tokens=384,
                       parse_fail_dir=None):
    model.eval()
    tokenizer.padding_side = "left"
    device     = next(model.parameters()).device
    image_root = Path(image_root)
    sbert      = _load_sbert(sbert_path, device)

    # ── prepare ──────────────────────────────────────────────────────────
    prepared = []
    for sample in test_data:
        gt_label  = assistant_label(sample) or {}
        img_path  = image_root / Path(sample["image"]).name
        image     = Image.open(img_path).convert("RGB")
        user_text = next(
            (item["text"] for msg in sample["messages"] if msg["role"] == "user"
             for item in msg["content"] if item.get("type") == "text"), "")

        pixel_values    = transform(image).unsqueeze(0).to(torch.float16)
        img_placeholder = "<img>" + img_context_token * num_visual_tokens + "</img>"
        prompt = (f"<|system|>\n{system_prompt}</s>"
                  f"<|user|>\n{img_placeholder}\n{user_text}</s>"
                  f"<|assistant|>\n")
        prepared.append({"sample": sample, "gt_label": gt_label,
                          "pixel_values": pixel_values, "prompt": prompt})

    # ── batched inference ────────────────────────────────────────────────
    records = []
    n_parse_ok = level_correct = 0
    tp = fp = fn = 0
    gt_descs, pred_descs = [], []
    lvl_tp = {lv: 0 for lv in LEVELS}
    lvl_fp = {lv: 0 for lv in LEVELS}
    lvl_fn = {lv: 0 for lv in LEVELS}

    for start in tqdm(range(0, len(prepared), eval_batch_size), desc=desc):
        items = prepared[start: start + eval_batch_size]

        pixel_values = torch.cat(
            [it["pixel_values"] for it in items], dim=0).to(device)

        # left-pad tokenised prompts to same length
        encs = [tokenizer(it["prompt"], return_tensors="pt",
                          add_special_tokens=False) for it in items]
        max_len        = max(e["input_ids"].shape[1] for e in encs)
        input_ids      = torch.zeros(len(items), max_len, dtype=torch.long).to(device)
        attention_mask = torch.zeros(len(items), max_len, dtype=torch.long).to(device)
        for i, enc in enumerate(encs):
            seq_len = enc["input_ids"].shape[1]
            input_ids[i, -seq_len:]      = enc["input_ids"][0]
            attention_mask[i, -seq_len:] = enc["attention_mask"][0]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            gen = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
            )

        for i, item in enumerate(items):
            # Decode the full output sequence — InternVL2's tokenizer with
            # skip_special_tokens=True automatically strips the prompt's
            # special tokens (<|system|>, <|user|>, img tokens, etc.),
            # leaving only the assistant's generated text.
            out_text = tokenizer.decode(gen[i], skip_special_tokens=True).strip()
            n_parse_ok, level_correct, tp, fp, fn = _accumulate(
                records, item["sample"], item["gt_label"], out_text,
                n_parse_ok, level_correct, tp, fp, fn, gt_descs, pred_descs,
                lvl_tp, lvl_fp, lvl_fn)

    return _finalise(records, len(test_data), n_parse_ok, level_correct,
                     tp, fp, fn, gt_descs, pred_descs, sbert, device,
                     lvl_tp, lvl_fp, lvl_fn,
                     image_root=image_root, parse_fail_dir=parse_fail_dir, desc=desc)

# ---------------------------------------------------------------------------
# OpenAI GPT-4V (or any vision model via OpenAI API) — one request per sample
# ---------------------------------------------------------------------------
def evaluate_openai(test_data, image_root,
                    system_prompt, desc="eval",
                    model_name="gpt-4o",
                    max_tokens=512,
                    sbert_path=None,
                    parse_fail_dir=None,
                    requests_per_minute=20):
    """Evaluate any OpenAI vision model on InspecSafe (level classification task).

    Requires:  pip install openai
    Auth:      set OPENAI_API_KEY environment variable.
    Returns the same dict schema as evaluate_qwen25vl / evaluate_gemma3.
    """
    import base64, time
    from openai import OpenAI

    client     = OpenAI()
    image_root = Path(image_root)
    delay      = 60.0 / requests_per_minute

    # optional SBERT (no GPU object available, use CPU)
    sbert = None
    if sbert_path and os.path.isdir(sbert_path):
        try:
            from sentence_transformers import SentenceTransformer
            import torch as _torch
            sbert = SentenceTransformer(sbert_path, device="cpu")
        except Exception as e:
            print(f"  (SBERT unavailable: {e})")

    records = []
    n_parse_ok = level_correct = 0
    tp = fp = fn = 0
    gt_descs, pred_descs = [], []
    lvl_tp = {lv: 0 for lv in LEVELS}
    lvl_fp = {lv: 0 for lv in LEVELS}
    lvl_fn = {lv: 0 for lv in LEVELS}

    for sample in tqdm(test_data, desc=desc):
        gt_label  = assistant_label(sample) or {}
        img_path  = image_root / Path(sample["image"]).name
        user_text = next(
            (item["text"] for msg in sample["messages"] if msg["role"] == "user"
             for item in msg["content"] if item.get("type") == "text"), "")

        # encode image as base64
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        suffix = img_path.suffix.lower().lstrip(".")
        mime   = f"image/{'jpeg' if suffix in ('jpg','jpeg') else suffix}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{img_b64}", "detail": "high"}},
                {"type": "text", "text": user_text},
            ]},
        ]

        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages,
                max_tokens=max_tokens, temperature=0,
            )
            out_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error on {img_path.name}: {e}")
            out_text = ""

        n_parse_ok, level_correct, tp, fp, fn = _accumulate(
            records, sample, gt_label, out_text,
            n_parse_ok, level_correct, tp, fp, fn, gt_descs, pred_descs,
            lvl_tp, lvl_fp, lvl_fn)

        time.sleep(delay)

    device = "cpu"
    return _finalise(records, len(test_data), n_parse_ok, level_correct,
                     tp, fp, fn, gt_descs, pred_descs, sbert, device,
                     lvl_tp, lvl_fp, lvl_fn,
                     image_root=image_root, parse_fail_dir=parse_fail_dir, desc=desc)

# ---------------------------------------------------------------------------
# Google Gemini (gemini-1.5-flash or any vision model) — one request per sample
# ---------------------------------------------------------------------------
def evaluate_gemini(test_data, image_root,
                    system_prompt, credentials_json,
                    desc="eval",
                    model_name="gemini-1.5-flash",
                    max_tokens=512,
                    sbert_path=None,
                    parse_fail_dir=None,
                    requests_per_minute=15):
    """Evaluate a Gemini vision model on InspecSafe (level classification task).

    Requires:  pip install google-generativeai google-auth
    Auth:      pass path to GCP service account JSON key file.
    Returns the same dict schema as evaluate_qwen25vl / evaluate_gemma3.
    """
    import time
    import google.generativeai as genai
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        credentials_json,
        scopes=["https://www.googleapis.com/auth/generative-language"])
    genai.configure(credentials=credentials)
    model      = genai.GenerativeModel(model_name=model_name,
                                       system_instruction=system_prompt)
    image_root = Path(image_root)
    delay      = 60.0 / requests_per_minute

    sbert = None
    if sbert_path and os.path.isdir(sbert_path):
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer(sbert_path, device="cpu")
        except Exception as e:
            print(f"  (SBERT unavailable: {e})")

    records = []
    n_parse_ok = level_correct = 0
    tp = fp = fn = 0
    gt_descs, pred_descs = [], []
    lvl_tp = {lv: 0 for lv in LEVELS}
    lvl_fp = {lv: 0 for lv in LEVELS}
    lvl_fn = {lv: 0 for lv in LEVELS}

    for sample in tqdm(test_data, desc=desc):
        gt_label  = assistant_label(sample) or {}
        img_path  = image_root / Path(sample["image"]).name
        user_text = next(
            (item["text"] for msg in sample["messages"] if msg["role"] == "user"
             for item in msg["content"] if item.get("type") == "text"), "")

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        suffix = img_path.suffix.lower().lstrip(".")
        mime   = f"image/{'jpeg' if suffix in ('jpg','jpeg') else suffix}"

        try:
            response = model.generate_content(
                [{"mime_type": mime, "data": img_bytes}, user_text],
                generation_config={"max_output_tokens": max_tokens, "temperature": 0},
            )
            out_text = response.text.strip()
        except Exception as e:
            print(f"  API error on {img_path.name}: {e}")
            out_text = ""

        n_parse_ok, level_correct, tp, fp, fn = _accumulate(
            records, sample, gt_label, out_text,
            n_parse_ok, level_correct, tp, fp, fn, gt_descs, pred_descs,
            lvl_tp, lvl_fp, lvl_fn)

        time.sleep(delay)

    device = "cpu"
    return _finalise(records, len(test_data), n_parse_ok, level_correct,
                     tp, fp, fn, gt_descs, pred_descs, sbert, device,
                     lvl_tp, lvl_fp, lvl_fn,
                     image_root=image_root, parse_fail_dir=parse_fail_dir, desc=desc)
