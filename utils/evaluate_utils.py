"""Utilities for evaluating saved VLM result JSON files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CONSTRUCTIONSITE10K_SBERT_PATH = "/root/autodl-tmp/all-MiniLM-L6-v2"
INSPECSAFE_SAFETY_LEVELS = ["Level I", "Level II", "Level III", "Level IV"]


def extract_label(output: str | None) -> str | None:
    """Extract safe/unsafe label from VLM output text."""
    if not output:
        return None

    text = output.strip().lower()
    match = re.search(r"final\s+label\s*:\s*(safe|unsafe)", text)
    if match:
        return match.group(1)

    matches = re.findall(r"\b(unsafe|safe)\b", text)
    if matches:
        return matches[-1]

    return None


def extract_choice_label(output: str | None) -> str | None:
    """Extract an A-D multiple-choice answer from VLM output text."""
    if not output:
        return None

    text = output.strip().upper()
    match = re.match(r"^\s*([ABCD])\b", text)
    if match:
        return match.group(1)

    match = re.search(r"\b(?:ANSWER|OPTION|FINAL)\s*(?:ANSWER)?\s*[:\-]?\s*([ABCD])\b", text)
    if match:
        return match.group(1)

    matches = re.findall(r"\b([ABCD])\b", text)
    if matches:
        return matches[-1]

    return None


def _normalize_hazard_label(label: str) -> str | None:
    normalized = label.strip().lower().replace("_", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in {"hazardous", "unsafe"}:
        return "hazardous"
    if normalized in {"non-hazardous", "non hazardous", "not hazardous", "safe"}:
        return "non-hazardous"
    return None


def extract_hazard_label(output: str | None) -> str | None:
    """Extract hazardous/non-hazardous label from VLM output text."""
    if not output:
        return None

    text = output.strip().lower()
    final_match = re.search(
        r"final\s+label\s*:\s*(non[\s-]?hazardous|not\s+hazardous|hazardous|unsafe|safe)",
        text,
    )
    if final_match:
        return _normalize_hazard_label(final_match.group(1))

    matches = [
        _normalize_hazard_label(match.group(1))
        for match in re.finditer(
            r"\b(non[\s-]?hazardous|not\s+hazardous|safe|hazardous|unsafe)\b",
            text,
        )
    ]
    matches = [match for match in matches if match is not None]
    return matches[-1] if matches else None


def evaluate_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved inference results and optionally write updated JSON.

    The input can be a path to a JSON file, a top-level JSON object containing a
    ``results`` list, or a raw list of sample result dictionaries. Each sample is
    updated with ``predicted`` and ``status`` fields.
    """
    source_path: Path | None = None
    if isinstance(results_json, (str, Path)):
        source_path = Path(results_json)
        with source_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] | list[dict[str, Any]] = json.load(file)
    else:
        payload = results_json

    if isinstance(payload, list):
        data: dict[str, Any] = {"results": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("results_json must be a path, dict, or list of dictionaries.")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Results JSON must contain a 'results' list.")

    total = len(results)
    correct = 0
    errors = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for sample in results:
        if not isinstance(sample, dict):
            errors += 1
            continue

        ground_truth = str(sample.get("ground_truth", "")).strip().lower()
        sample["ground_truth"] = ground_truth

        if sample.get("error"):
            sample["predicted"] = None
            sample["status"] = "ERROR"
            errors += 1
            continue

        if ground_truth not in ("safe", "unsafe"):
            sample["predicted"] = None
            sample["status"] = "SKIP"
            errors += 1
            continue

        predicted = extract_label(sample.get("output"))
        sample["predicted"] = predicted

        if predicted is None:
            sample["status"] = "PARSE_FAIL"
            errors += 1
        elif predicted == ground_truth:
            sample["status"] = "CORRECT"
            correct += 1
            if predicted == "unsafe":
                tp += 1
            else:
                tn += 1
        else:
            sample["status"] = "WRONG"
            if predicted == "unsafe":
                fp += 1
            else:
                fn += 1

    evaluated = total - errors
    data["summary"] = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "errors_or_skipped": errors,
        "accuracy": correct / evaluated if evaluated > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    if output_json is not None:
        target_path = Path(output_json)
    else:
        target_path = source_path

    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    return data


def extract_inspecsafe_safety_level_json(
    output: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract the first complete JSON object from a safety-level response."""
    if isinstance(output, dict):
        return output
    if not output:
        return None

    text = str(output)
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for index, character in enumerate(text[start:], start=start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth != 0:
                continue

            blob = text[start : index + 1]
            try:
                parsed = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                repaired = blob.replace("'", '"')
                repaired = re.sub(r",\s*}", "}", repaired)
                repaired = re.sub(r",\s*]", "]", repaired)
                try:
                    parsed = json.loads(repaired)
                except (json.JSONDecodeError, TypeError):
                    return None
            return parsed if isinstance(parsed, dict) else None

    return None


def normalize_inspecsafe_safety_level(value: Any) -> str | None:
    """Normalize Roman numerals, numbers, or words to ``Level I``-``IV``."""
    if value is None or str(value).strip() == "":
        return None

    match = re.search(
        r"(IV|III|II|I|[1-4]|one|two|three|four)",
        str(value),
        re.IGNORECASE,
    )
    if not match:
        return None

    token = match.group(1).lower()
    return {
        "i": "Level I",
        "1": "Level I",
        "one": "Level I",
        "ii": "Level II",
        "2": "Level II",
        "two": "Level II",
        "iii": "Level III",
        "3": "Level III",
        "three": "Level III",
        "iv": "Level IV",
        "4": "Level IV",
        "four": "Level IV",
    }.get(token)


def inspecsafe_hazard_set(label: dict[str, Any] | None) -> set[str]:
    """Return normalized hazard phrases from a parsed InspecSafe label."""
    if not label:
        return set()
    hazards = label.get("hazards", [])
    if not isinstance(hazards, list):
        return set()
    return {str(hazard).strip().lower() for hazard in hazards if str(hazard).strip()}


def _inspecsafe_ground_truth(sample: dict[str, Any]) -> dict[str, Any]:
    for key in ("ground_truth", "ground_truth_output", "gt"):
        parsed = extract_inspecsafe_safety_level_json(sample.get(key))
        if parsed is not None:
            return parsed
    return {}


def _scene_sbert_similarity(
    ground_truth_descriptions: list[str],
    predicted_descriptions: list[str],
    sbert_path: str | Path | None,
) -> float | None:
    if not sbert_path or not Path(sbert_path).is_dir() or not ground_truth_descriptions:
        return None

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(sbert_path))
        ground_truth_embeddings = model.encode(
            ground_truth_descriptions,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        predicted_embeddings = model.encode(
            predicted_descriptions,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        similarities = torch.nn.functional.cosine_similarity(
            ground_truth_embeddings,
            predicted_embeddings,
        )
        return float(similarities.mean().cpu().item())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"  (SBERT unavailable: {exc})")
        return None


def evaluate_inspecsafe_safety_level_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
    *,
    compute_scene_metrics: bool = True,
    sbert_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score InspecSafe four-level JSON outputs with the reference metrics.

    Metrics intentionally match ``.plan/reference/inspecsafe_eval_utils.py``:
    JSON parse rate, level accuracy and per-level/macro/micro P/R/F1, hazard
    micro P/R/F1, and optional scene-description SBERT similarity.
    """
    source_path: Path | None = None
    if isinstance(results_json, (str, Path)):
        source_path = Path(results_json)
        with source_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] | list[dict[str, Any]] = json.load(file)
    else:
        payload = results_json

    if isinstance(payload, list):
        data: dict[str, Any] = {"results": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("results_json must be a path, dict, or list of dictionaries.")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Results JSON must contain a 'results' list.")

    level_tp = {level: 0 for level in INSPECSAFE_SAFETY_LEVELS}
    level_fp = {level: 0 for level in INSPECSAFE_SAFETY_LEVELS}
    level_fn = {level: 0 for level in INSPECSAFE_SAFETY_LEVELS}
    level_counts = {level: 0 for level in INSPECSAFE_SAFETY_LEVELS}
    level_correct_counts = {level: 0 for level in INSPECSAFE_SAFETY_LEVELS}
    parse_ok_count = 0
    level_correct = 0
    hazard_tp = hazard_fp = hazard_fn = 0
    ground_truth_descriptions: list[str] = []
    predicted_descriptions: list[str] = []

    for sample in results:
        if not isinstance(sample, dict):
            ground_truth_descriptions.append("")
            predicted_descriptions.append("")
            continue

        ground_truth = _inspecsafe_ground_truth(sample)
        prediction = (
            None
            if sample.get("error")
            else extract_inspecsafe_safety_level_json(
                sample.get("output") or sample.get("raw_output")
            )
        )
        if prediction is not None:
            parse_ok_count += 1

        ground_truth_level = normalize_inspecsafe_safety_level(
            ground_truth.get("overall_safety_level")
        )
        predicted_level = normalize_inspecsafe_safety_level(
            prediction.get("overall_safety_level") if prediction else None
        )
        sample["ground_truth"] = ground_truth
        sample["predicted"] = prediction
        sample["gt_level"] = ground_truth_level
        sample["pred_level"] = predicted_level
        sample["parse_failed"] = prediction is None

        if ground_truth_level in level_counts:
            level_counts[ground_truth_level] += 1
        if ground_truth_level is not None and ground_truth_level == predicted_level:
            level_correct += 1
            level_correct_counts[ground_truth_level] += 1

        for level in INSPECSAFE_SAFETY_LEVELS:
            ground_truth_positive = ground_truth_level == level
            predicted_positive = predicted_level == level
            if ground_truth_positive and predicted_positive:
                level_tp[level] += 1
            elif predicted_positive and not ground_truth_positive:
                level_fp[level] += 1
            elif ground_truth_positive and not predicted_positive:
                level_fn[level] += 1

        ground_truth_hazards = inspecsafe_hazard_set(ground_truth)
        predicted_hazards = inspecsafe_hazard_set(prediction)
        hazard_tp += len(ground_truth_hazards & predicted_hazards)
        hazard_fp += len(predicted_hazards - ground_truth_hazards)
        hazard_fn += len(ground_truth_hazards - predicted_hazards)

        ground_truth_descriptions.append(str(ground_truth.get("scene_description", "")))
        predicted_descriptions.append(
            str(prediction.get("scene_description", "")) if prediction else ""
        )

        if sample.get("error"):
            sample["status"] = "ERROR"
        elif prediction is None:
            sample["status"] = "PARSE_FAIL"
        elif ground_truth_level is None:
            sample["status"] = "SKIP"
        elif predicted_level == ground_truth_level:
            sample["status"] = "CORRECT"
        else:
            sample["status"] = "WRONG"

    per_level: dict[str, dict[str, int | float]] = {}
    for level in INSPECSAFE_SAFETY_LEVELS:
        true_positive = level_tp[level]
        false_positive = level_fp[level]
        false_negative = level_fn[level]
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        count = level_counts[level]
        per_level[level] = {
            "n": count,
            "acc": level_correct_counts[level] / count if count else 0.0,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    active_levels = [
        level for level in INSPECSAFE_SAFETY_LEVELS if level_counts[level] > 0
    ]
    active_count = len(active_levels)
    macro_precision = (
        sum(float(per_level[level]["precision"]) for level in active_levels)
        / active_count
        if active_count
        else 0.0
    )
    macro_recall = (
        sum(float(per_level[level]["recall"]) for level in active_levels)
        / active_count
        if active_count
        else 0.0
    )
    macro_f1 = (
        sum(float(per_level[level]["f1"]) for level in active_levels) / active_count
        if active_count
        else 0.0
    )

    total_level_tp = sum(level_tp.values())
    total_level_fp = sum(level_fp.values())
    total_level_fn = sum(level_fn.values())
    micro_precision = (
        total_level_tp / (total_level_tp + total_level_fp)
        if total_level_tp + total_level_fp
        else 0.0
    )
    micro_recall = (
        total_level_tp / (total_level_tp + total_level_fn)
        if total_level_tp + total_level_fn
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    hazard_precision = (
        hazard_tp / (hazard_tp + hazard_fp) if hazard_tp + hazard_fp else 0.0
    )
    hazard_recall = (
        hazard_tp / (hazard_tp + hazard_fn) if hazard_tp + hazard_fn else 0.0
    )
    hazard_f1 = (
        2 * hazard_precision * hazard_recall / (hazard_precision + hazard_recall)
        if hazard_precision + hazard_recall
        else 0.0
    )

    sample_count = len(results)
    scene_sbert_sim = (
        _scene_sbert_similarity(
            ground_truth_descriptions,
            predicted_descriptions,
            sbert_path,
        )
        if compute_scene_metrics
        else None
    )
    data["summary"] = {
        "n_samples": sample_count,
        "json_parse_rate": parse_ok_count / sample_count if sample_count else 0.0,
        "level_accuracy": level_correct / sample_count if sample_count else 0.0,
        "per_level": per_level,
        "level_macro_p": macro_precision,
        "level_macro_r": macro_recall,
        "level_macro_f1": macro_f1,
        "level_micro_p": micro_precision,
        "level_micro_r": micro_recall,
        "level_micro_f1": micro_f1,
        "hazard_precision": hazard_precision,
        "hazard_recall": hazard_recall,
        "hazard_f1": hazard_f1,
        "scene_sbert_sim": scene_sbert_sim,
    }

    target_path = Path(output_json) if output_json is not None else source_path
    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    return data


def evaluate_labsafety_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved Lab Safety multiple-choice inference results."""
    source_path: Path | None = None
    if isinstance(results_json, (str, Path)):
        source_path = Path(results_json)
        with source_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] | list[dict[str, Any]] = json.load(file)
    else:
        payload = results_json

    if isinstance(payload, list):
        data: dict[str, Any] = {"results": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("results_json must be a path, dict, or list of dictionaries.")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Results JSON must contain a 'results' list.")

    labels = ["A", "B", "C", "D"]
    confusion = {
        truth: {pred: 0 for pred in [*labels, "PARSE_FAIL"]}
        for truth in labels
    }
    total = len(results)
    correct = 0
    errors = 0
    parse_failures = 0

    for sample in results:
        if not isinstance(sample, dict):
            errors += 1
            continue

        ground_truth = str(
            sample.get("ground_truth_answer") or sample.get("ground_truth") or ""
        ).strip().upper()
        sample["ground_truth_answer"] = ground_truth

        if sample.get("error"):
            sample["predicted"] = None
            sample["status"] = "ERROR"
            errors += 1
            if ground_truth in confusion:
                confusion[ground_truth]["PARSE_FAIL"] += 1
            continue

        if ground_truth not in labels:
            sample["predicted"] = None
            sample["status"] = "SKIP"
            errors += 1
            continue

        predicted = extract_choice_label(sample.get("output"))
        sample["predicted"] = predicted

        if predicted is None:
            sample["status"] = "PARSE_FAIL"
            parse_failures += 1
            errors += 1
            confusion[ground_truth]["PARSE_FAIL"] += 1
        elif predicted == ground_truth:
            sample["status"] = "CORRECT"
            correct += 1
            confusion[ground_truth][predicted] += 1
        else:
            sample["status"] = "WRONG"
            confusion[ground_truth][predicted] += 1

    evaluated = total - errors
    data["summary"] = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "errors_or_skipped": errors,
        "parse_failures": parse_failures,
        "accuracy": correct / evaluated if evaluated > 0 else 0.0,
    }
    data["confusion"] = confusion

    if output_json is not None:
        target_path = Path(output_json)
    else:
        target_path = source_path

    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    return data


def evaluate_labsafety_gen_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved LabSafety-v1 hazardous/non-hazardous inference results."""
    source_path: Path | None = None
    if isinstance(results_json, (str, Path)):
        source_path = Path(results_json)
        with source_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] | list[dict[str, Any]] = json.load(file)
    else:
        payload = results_json

    if isinstance(payload, list):
        data: dict[str, Any] = {"results": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("results_json must be a path, dict, or list of dictionaries.")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Results JSON must contain a 'results' list.")

    labels = ["hazardous", "non-hazardous"]
    confusion = {truth: {pred: 0 for pred in [*labels, "PARSE_FAIL"]} for truth in labels}
    total = len(results)
    correct = 0
    errors = 0
    parse_failures = 0
    tp = fp = tn = fn = 0

    for sample in results:
        if not isinstance(sample, dict):
            errors += 1
            continue

        ground_truth = _normalize_hazard_label(
            str(
                sample.get("ground_truth_hazard_label")
                or sample.get("ground_truth_label")
                or sample.get("ground_truth")
                or ""
            )
        )
        sample["ground_truth_hazard_label"] = ground_truth

        if sample.get("error"):
            sample["predicted"] = None
            sample["status"] = "ERROR"
            errors += 1
            if ground_truth in confusion:
                confusion[ground_truth]["PARSE_FAIL"] += 1
            continue

        if ground_truth not in labels:
            sample["predicted"] = None
            sample["status"] = "SKIP"
            errors += 1
            continue

        predicted = extract_hazard_label(sample.get("output"))
        sample["predicted"] = predicted

        if predicted is None:
            sample["status"] = "PARSE_FAIL"
            parse_failures += 1
            errors += 1
            confusion[ground_truth]["PARSE_FAIL"] += 1
        elif predicted == ground_truth:
            sample["status"] = "CORRECT"
            correct += 1
            confusion[ground_truth][predicted] += 1
            if predicted == "hazardous":
                tp += 1
            else:
                tn += 1
        else:
            sample["status"] = "WRONG"
            confusion[ground_truth][predicted] += 1
            if predicted == "hazardous":
                fp += 1
            else:
                fn += 1

    evaluated = total - errors
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    data["summary"] = {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "errors_or_skipped": errors,
        "parse_failures": parse_failures,
        "accuracy": correct / evaluated if evaluated > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "hazardous_precision": precision,
        "hazardous_recall": recall,
        "hazardous_f1": f1,
    }
    data["confusion"] = confusion

    if output_json is not None:
        target_path = Path(output_json)
    else:
        target_path = source_path

    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    return data


def parse_constructionsite10k_output(text: str | None) -> tuple[dict[str, Any], bool]:
    """Parse a ConstructionSite-10K JSON response from model text.

    This intentionally mirrors ``constructionsite_10k/evaluate_utils.py`` so
    saved Image_RAG outputs are scored with the same semantics as the
    fine-tuned-model evaluation scripts.
    """
    if not text:
        return {"annotation": "", "violations": []}, False

    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and "violations" in parsed:
            return parsed, True
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict) and "violations" in parsed:
                    return parsed, True
            except Exception:
                pass

    return {"annotation": "", "violations": []}, False


def get_violation_rules(violations: Any) -> set[int]:
    rules: set[int] = set()
    if not isinstance(violations, list):
        return rules

    for violation in violations:
        if not isinstance(violation, dict):
            continue
        try:
            rule = int(violation.get("rule"))
        except (TypeError, ValueError):
            continue
        rules.add(rule)
    return rules


def _load_annotation_scorers(sbert_path: str | Path | None) -> tuple[Any, Any, Any]:
    """Load ROUGE-L and SBERT scorers used by the ConstructionSite-10K scripts."""
    try:
        from rouge_score import rouge_scorer
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as st_util
    except ImportError as exc:
        raise ImportError(
            "ConstructionSite-10K annotation metrics require 'rouge-score' and "
            "'sentence-transformers'. Install them or pass "
            "compute_annotation_metrics=False."
        ) from exc

    if not sbert_path:
        raise ValueError("sbert_path is required when computing SBERT similarity.")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sbert = SentenceTransformer(str(sbert_path))
    return rouge, sbert, st_util


def evaluate_constructionsite10k_results_json(
    results_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_json: str | Path | None = None,
    *,
    compute_annotation_metrics: bool = False,
    sbert_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved ConstructionSite-10K inference results."""
    source_path: Path | None = None
    if isinstance(results_json, (str, Path)):
        source_path = Path(results_json)
        with source_path.open("r", encoding="utf-8") as file:
            payload: dict[str, Any] | list[dict[str, Any]] = json.load(file)
    else:
        payload = results_json

    if isinstance(payload, list):
        data: dict[str, Any] = {"results": payload}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("results_json must be a path, dict, or list of dictionaries.")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Results JSON must contain a 'results' list.")

    all_rules = [1, 2, 3, 4]
    tp = {rule: 0 for rule in all_rules}
    fp = {rule: 0 for rule in all_rules}
    fn = {rule: 0 for rule in all_rules}
    parse_failures = 0
    valid_results: list[dict[str, Any]] = []
    rouge_scores: list[float] = []
    sbert_scores: list[float] = []
    rouge = sbert = st_util = None
    if compute_annotation_metrics:
        rouge, sbert, st_util = _load_annotation_scorers(sbert_path)

    for sample in results:
        if not isinstance(sample, dict):
            parse_failures += 1
            continue

        gt_text = sample.get("ground_truth_output") or sample.get("ground_truth") or ""
        pred_text = sample.get("output") or sample.get("pred_raw") or ""

        gt_json, _ = parse_constructionsite10k_output(gt_text)
        gt_rules = get_violation_rules(gt_json.get("violations", []))

        sample["gt_annotation"] = gt_json.get("annotation", "")
        sample["gt_rules"] = sorted(gt_rules)

        if sample.get("error"):
            sample["pred_annotation"] = ""
            sample["pred_rules"] = []
            if compute_annotation_metrics:
                sample["rouge_l"] = 0.0
                sample["sbert_sim"] = 0.0
            sample["parse_failed"] = True
            sample["status"] = "ERROR"
            parse_failures += 1
            continue

        pred_json, parse_ok = parse_constructionsite10k_output(pred_text)
        pred_rules = get_violation_rules(pred_json.get("violations", []))

        sample["pred_annotation"] = pred_json.get("annotation", "") if parse_ok else ""
        sample["pred_rules"] = sorted(pred_rules) if parse_ok else []
        sample["parse_failed"] = not parse_ok

        if not parse_ok:
            if compute_annotation_metrics:
                sample["rouge_l"] = 0.0
                sample["sbert_sim"] = 0.0
            sample["status"] = "PARSE_FAIL"
            parse_failures += 1
            continue

        for rule in all_rules:
            pred_pos = rule in pred_rules
            gt_pos = rule in gt_rules
            if pred_pos and gt_pos:
                tp[rule] += 1
            elif pred_pos and not gt_pos:
                fp[rule] += 1
            elif not pred_pos and gt_pos:
                fn[rule] += 1

        if compute_annotation_metrics:
            gt_annotation = str(gt_json.get("annotation", ""))
            pred_annotation = str(pred_json.get("annotation", ""))
            rouge_l = rouge.score(gt_annotation, pred_annotation)["rougeL"].fmeasure
            rouge_scores.append(rouge_l)
            sbert_sim = st_util.cos_sim(
                sbert.encode(pred_annotation, convert_to_tensor=True),
                sbert.encode(gt_annotation, convert_to_tensor=True),
            ).item()
            sbert_scores.append(sbert_sim)
            sample["rouge_l"] = rouge_l
            sample["sbert_sim"] = sbert_sim

        sample["status"] = "CORRECT" if pred_rules == gt_rules else "WRONG"
        valid_results.append(sample)

    per_rule = {}
    for rule in all_rules:
        precision = tp[rule] / (tp[rule] + fp[rule]) if tp[rule] + fp[rule] else 0.0
        recall = tp[rule] / (tp[rule] + fn[rule]) if tp[rule] + fn[rule] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_rule[str(rule)] = {
            "tp": tp[rule],
            "fp": fp[rule],
            "fn": fn[rule],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    valid_count = len(valid_results)
    exact_match = sum(
        1 for sample in valid_results if sample["gt_rules"] == sample["pred_rules"]
    )
    safe_correct = sum(
        1
        for sample in valid_results
        if (len(sample["gt_rules"]) == 0) == (len(sample["pred_rules"]) == 0)
    )
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    data["summary"] = {
        "total_samples": len(results),
        "valid_samples": valid_count,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(results) if results else 0.0,
        "exact_match_acc": exact_match / valid_count if valid_count else 0.0,
        "safe_unsafe_acc": safe_correct / valid_count if valid_count else 0.0,
        "macro_precision": sum(per_rule[str(r)]["precision"] for r in all_rules)
        / len(all_rules),
        "macro_recall": sum(per_rule[str(r)]["recall"] for r in all_rules)
        / len(all_rules),
        "macro_f1": sum(per_rule[str(r)]["f1"] for r in all_rules) / len(all_rules),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "avg_rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "avg_sbert_sim": sum(sbert_scores) / len(sbert_scores) if sbert_scores else 0.0,
    }
    data["per_rule"] = per_rule
    data["details"] = results

    if output_json is not None:
        target_path = Path(output_json)
    else:
        target_path = source_path

    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved result JSON.")
    parser.add_argument("results_json", type=Path, help="Path to saved results JSON.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the evaluated JSON. Defaults to overwriting input.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=[
            "auto",
            "inspecsafe",
            "inspecsafe_safety_level",
            "constructionsite10k",
            "lab_safety",
            "lab_safety_gen",
        ],
        default="auto",
        help="Evaluation metric type for the saved JSON.",
    )
    parser.add_argument(
        "--sbert-path",
        type=Path,
        default=Path(DEFAULT_CONSTRUCTIONSITE10K_SBERT_PATH),
        help=(
            "SentenceTransformer path for ConstructionSite-10K annotation "
            "similarity metrics."
        ),
    )
    parser.add_argument(
        "--skip-annotation-metrics",
        action="store_true",
        help="Skip ROUGE-L and SBERT metrics for ConstructionSite-10K.",
    )
    parser.add_argument(
        "--skip-scene-metrics",
        action="store_true",
        help="Skip SBERT scene-description similarity for InspecSafe safety levels.",
    )
    return parser.parse_args()


def _detect_results_type(results_json: Path) -> str:
    with results_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("task_type") == "safety level":
            return "inspecsafe_safety_level"
    results = payload.get("results", []) if isinstance(payload, dict) else payload
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            if "ground_truth_hazard_label" in first:
                return "lab_safety_gen"
            ground_truth = first.get("ground_truth")
            if isinstance(ground_truth, dict) and "overall_safety_level" in ground_truth:
                return "inspecsafe_safety_level"
            if "ground_truth_answer" in first:
                return "lab_safety"
            if "ground_truth_output" in first:
                return "constructionsite10k"
    return "inspecsafe"


if __name__ == "__main__":
    args = _parse_args()
    dataset_type = (
        _detect_results_type(args.results_json)
        if args.dataset_type == "auto"
        else args.dataset_type
    )
    if dataset_type == "constructionsite10k":
        evaluated = evaluate_constructionsite10k_results_json(
            args.results_json,
            args.output_json,
            compute_annotation_metrics=not args.skip_annotation_metrics,
            sbert_path=args.sbert_path,
        )
        summary = evaluated["summary"]
        print(f"Total samples:  {summary['total_samples']}")
        print(f"Valid samples:  {summary['valid_samples']}")
        print(f"Parse failures: {summary['parse_failures']}")
        print(f"Exact match:    {summary['exact_match_acc']:.4f}")
        print(f"Safe/unsafe:    {summary['safe_unsafe_acc']:.4f}")
        print(f"Macro Precision:{summary['macro_precision']:.4f}")
        print(f"Macro Recall:   {summary['macro_recall']:.4f}")
        print(f"Macro F1:       {summary['macro_f1']:.4f}")
        print(f"Micro Precision:{summary['micro_precision']:.4f}")
        print(f"Micro Recall:   {summary['micro_recall']:.4f}")
        print(f"Micro F1:       {summary['micro_f1']:.4f}")
        print(f"Avg ROUGE-L:    {summary['avg_rouge_l']:.4f}")
        print(f"Avg SBERT sim:  {summary['avg_sbert_sim']:.4f}")
    elif dataset_type == "inspecsafe_safety_level":
        evaluated = evaluate_inspecsafe_safety_level_results_json(
            args.results_json,
            args.output_json,
            compute_scene_metrics=not args.skip_scene_metrics,
            sbert_path=args.sbert_path,
        )
        summary = evaluated["summary"]
        print(f"Samples:         {summary['n_samples']}")
        print(f"JSON parse rate: {summary['json_parse_rate']:.4f}")
        print(f"Level accuracy:  {summary['level_accuracy']:.4f}")
        print(f"Level macro F1:  {summary['level_macro_f1']:.4f}")
        print(f"Level micro F1:  {summary['level_micro_f1']:.4f}")
        print(f"Hazard F1:       {summary['hazard_f1']:.4f}")
        if summary["scene_sbert_sim"] is not None:
            print(f"Scene SBERT sim:  {summary['scene_sbert_sim']:.4f}")
    elif dataset_type == "lab_safety":
        evaluated = evaluate_labsafety_results_json(args.results_json, args.output_json)
        summary = evaluated["summary"]
        print(f"Total samples:  {summary['total']}")
        print(f"Evaluated:      {summary['evaluated']}")
        print(f"Correct:        {summary['correct']}")
        print(f"Errors/Skipped: {summary['errors_or_skipped']}")
        print(f"Parse failures: {summary['parse_failures']}")
        print(
            "Accuracy:       "
            f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
        )
    elif dataset_type == "lab_safety_gen":
        evaluated = evaluate_labsafety_gen_results_json(args.results_json, args.output_json)
        summary = evaluated["summary"]
        print(f"Total samples:  {summary['total']}")
        print(f"Evaluated:      {summary['evaluated']}")
        print(f"Correct:        {summary['correct']}")
        print(f"Errors/Skipped: {summary['errors_or_skipped']}")
        print(f"Parse failures: {summary['parse_failures']}")
        print(f"TP:             {summary['tp']}")
        print(f"FP:             {summary['fp']}")
        print(f"TN:             {summary['tn']}")
        print(f"FN:             {summary['fn']}")
        print(
            "Accuracy:       "
            f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
        )
    else:
        evaluated = evaluate_results_json(args.results_json, args.output_json)
        summary = evaluated["summary"]
        print(f"Total samples:  {summary['total']}")
        print(f"Evaluated:      {summary['evaluated']}")
        print(f"Correct:        {summary['correct']}")
        print(f"Errors/Skipped: {summary['errors_or_skipped']}")
        print(f"TP:             {summary['tp']}")
        print(f"FP:             {summary['fp']}")
        print(f"TN:             {summary['tn']}")
        print(f"FN:             {summary['fn']}")
        print("Confusion matrix (positive=unsafe):")
        print("                Pred unsafe  Pred safe")
        print(f"Truth unsafe    {summary['tp']:>11}  {summary['fn']:>9}")
        print(f"Truth safe      {summary['fp']:>11}  {summary['tn']:>9}")
        print(
            "Accuracy:       "
            f"{summary['accuracy']:.4f} ({summary['correct']}/{summary['evaluated']})"
        )
