"""Build safety reasoning prompts from retrieved historical examples."""

import re
from pathlib import Path
from typing import Any

from config import GATED_RAG, PROJECT_ROOT, TOP_K


INSPECSAFE_SAFETY_LEVEL_SYSTEM_PROMPT = """You are an industrial safety inspector reviewing footage from an autonomous inspection robot deployed at an oil and gas / petrochemical facility.

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


def _inspecsafe_reference_level(item: dict[str, Any]) -> str:
    explicit_level = (
        item.get("overall_safety_level")
        or item.get("safety_level")
        or item.get("level")
    )
    if explicit_level:
        return str(explicit_level)

    # Existing binary indexes still retain source paths such as ``Level01``.
    # Recover that annotation so they can support safety-level RAG immediately.
    match = re.search(
        r"level[\s_-]*0?([1-4])",
        str(item.get("image_path", "")),
        re.IGNORECASE,
    )
    if match:
        return {
            "1": "Level I",
            "2": "Level II",
            "3": "Level III",
            "4": "Level IV",
        }[match.group(1)]
    return str(item.get("safe_label", ""))


LAB_SAFETY_SYSTEM_PROMPT = """You are a laboratory safety expert reviewing images from a lab safety training dataset.

Carefully analyse the provided image and the multiple-choice question. Select the single best answer from the options given.

Read the question and all options carefully. Use visual evidence from the image to support your choice.
Output a single capital letter only: A, B, C, or D. No explanation, no punctuation."""


LAB_SAFETY_GEN_SYSTEM_PROMPT = """You are a laboratory safety expert reviewing synthetic laboratory-scene images.

Carefully inspect the query image for visible hazards, unsafe handling, missing PPE, unsafe storage, spills, fire or chemical risks, and other laboratory safety issues.

Use the retrieved examples only as reference cases. Classify the query image itself as exactly one of:
- hazardous
- non-hazardous

Return your answer in this format:
Query image observations:
Retrieved evidence:
Reasoning:
Final label: hazardous or non-hazardous"""


CONSTRUCTIONSITE10K_SYSTEM_PROMPT = """You are a professional construction site safety inspector with expertise in hazard identification and regulatory compliance.

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


def build_prompt(query: str, retrieved_items: list[dict[str, Any]]) -> str:
    examples = []
    for index, item in enumerate(retrieved_items, start=1):
        examples.append(
            "\n".join(
                [
                    f"Example {index}:",
                    f"Image path: {item['image_path']}",
                    f"Caption: {item['caption']}",
                    f"Historical safety label: {item['safe_label']}",
                ]
            )
        )

    context = "\n\n".join(examples) or "No similar historical cases were retrieved."
    return f"""You are a construction safety assistant.

User query:
{query}

Retrieved similar historical cases:
{context}

Task:
1. Summarize the retrieved evidence.
2. Decide whether the situation is likely safe or unsafe.
3. Explain the reasoning using the retrieved examples.
4. Return a final label: safe or unsafe.
"""


def format_retrieved_examples(retrieved_items: list[dict[str, Any]]) -> str:
    examples = []
    for index, item in enumerate(retrieved_items, start=1):
        distance = item.get("distance")
        distance_text = ""
        if distance is not None:
            distance_text = f"\nSimilarity distance: {float(distance):.4f}"
        examples.append(
            "\n".join(
                [
                    f"Example {index}:",
                    f"Image path: {item.get('image_path', '')}",
                    f"Caption: {item.get('caption', '')}",
                    f"Historical safety label: {item.get('safe_label', '')}",
                ]
            )
            + distance_text
        )

    return "\n\n".join(examples) or "No similar historical cases were retrieved."


def build_image_rag_prompt(
    query: str,
    retrieved_items: list[dict[str, Any]],
) -> str:
    context = format_retrieved_examples(retrieved_items)
    return f"""You are a construction safety visual inspection assistant.

Question for the query image:
{query}

Retrieved visually similar historical cases:
{context}

Use the query image as the primary evidence. Use the retrieved cases as reference
examples for likely hazards, normal conditions, and label consistency.

Return your answer in this format:
Query image observations:
Retrieved evidence:
Reasoning:
Final label: safe or unsafe
"""


def build_rag_messages(
    query: str,
    query_image_path: str | Path,
    retrieved_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build Qwen2.5-VL messages with retrieved images as proper content blocks.

    Each retrieved example is passed as an actual image followed by a text
    annotation, so the VLM can see the reference images rather than just reading
    file paths as text.
    """
    content: list[dict[str, str]] = []

    for i, item in enumerate(retrieved_items, 1):
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        content.append({"type": "image", "image": str(image_path)})
        content.append({
            "type": "text",
            "text": f"Reference {i}: {item['caption']} (label: {item['safe_label']})",
        })

    content.append({"type": "image", "image": str(query_image_path)})
    content.append({
        "type": "text",
        "text": (
            f"Query Image: {query}\n"
            "Classify ONLY this query image based on the reference examples above.\n\n"
            "Return your answer in this format:\n"
            "Query image observations:\n"
            "Retrieved evidence:\n"
            "Reasoning:\n"
            "Final label: safe or unsafe"
        ),
    })

    return [
        {
            "role": "system",
            "content": "You are a construction safety visual inspection assistant. "
            "Use the reference images to inform your judgement of the query image.",
        },
        {"role": "user", "content": content},
    ]


def build_inspecsafe_safety_level_rag_messages(
    query: str,
    query_image_path: str | Path,
    retrieved_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build multi-image InspecSafe messages for four-level safety scoring."""
    content: list[dict[str, str]] = []

    for index, item in enumerate(retrieved_items, start=1):
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        level = _inspecsafe_reference_level(item)
        hazards = item.get("hazards", "")
        reference_text = [
            f"Reference {index} scene: {item.get('scene_description') or item.get('caption', '')}",
            f"Reference label: {level}",
        ]
        if hazards:
            reference_text.append(f"Reference hazards: {hazards}")

        content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": "\n".join(reference_text)})

    content.append({"type": "image", "image": str(query_image_path)})
    content.append({
        "type": "text",
        "text": (
            f"Query image task: {query}\n"
            "Use the references only as supporting examples. Inspect and classify "
            "ONLY the query image. Return the required JSON object only."
        ),
    })

    return [
        {"role": "system", "content": INSPECSAFE_SAFETY_LEVEL_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_constructionsite10k_rag_messages(
    query: str,
    query_image_path: str | Path,
    retrieved_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build multi-image RAG messages for ConstructionSite-10K rule detection."""
    content: list[dict[str, str]] = []

    for index, item in enumerate(retrieved_items, start=1):
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        rules = item.get("violation_rules") or "none"
        annotation = item.get("caption", "")
        content.append({"type": "image", "image": str(image_path)})
        content.append({
            "type": "text",
            "text": (
                f"Reference {index}: {annotation}\n"
                f"Ground-truth violation rules: {rules}"
            ),
        })

    content.append({"type": "image", "image": str(query_image_path)})
    content.append({
        "type": "text",
        "text": (
            f"Query image task: {query}\n"
            "Use the reference examples for visual context only. "
            "Classify the query image under rules 1-4 and return JSON only."
        ),
    })

    return [
        {"role": "system", "content": CONSTRUCTIONSITE10K_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_labsafety_rag_messages(
    query: str,
    query_image_path: str | Path,
    retrieved_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build multi-image RAG messages for Lab Safety multiple-choice VQA."""
    content: list[dict[str, str]] = []

    for index, item in enumerate(retrieved_items, start=1):
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        question = item.get("question") or item.get("caption", "")
        answer = item.get("answer") or item.get("safe_label", "")
        explanation = item.get("explanation", "")
        category = item.get("category", "")
        level = item.get("level", "")

        details = [
            f"Reference {index}:",
            f"Question: {question}",
            f"Correct answer: {answer}",
        ]
        if explanation:
            details.append(f"Explanation: {explanation}")
        if category:
            details.append(f"Category: {category}")
        if level:
            details.append(f"Level: {level}")

        content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": "\n".join(details)})

    content.append({"type": "image", "image": str(query_image_path)})
    content.append({
        "type": "text",
        "text": (
            f"Query image question:\n{query}\n\n"
            "Use the reference examples only as lab-safety context. "
            "Answer the query image question with one capital letter only: A, B, C, or D."
        ),
    })

    return [
        {"role": "system", "content": LAB_SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_labsafety_gen_rag_messages(
    query: str,
    query_image_path: str | Path,
    retrieved_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build multi-image RAG messages for generated lab-safety classification."""
    content: list[dict[str, str]] = []

    for index, item in enumerate(retrieved_items, start=1):
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        label = item.get("safe_label", "")
        description = item.get("description") or item.get("caption", "")
        hazards = item.get("hazards", "")
        vlm_label = item.get("vlm_label", "")
        agree = item.get("agree", "")

        details = [
            f"Reference {index}:",
            f"Ground-truth label: {label}",
            f"Description: {description}",
        ]
        if hazards:
            details.append(f"Hazards: {hazards}")
        if vlm_label:
            details.append(f"VLM label check: {vlm_label}")
        if agree:
            details.append(f"Agreement flag: {agree}")

        content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": "\n".join(details)})

    content.append({"type": "image", "image": str(query_image_path)})
    content.append({
        "type": "text",
        "text": (
            f"Query image task: {query}\n"
            "Use the query image as primary evidence. Use the references only "
            "to calibrate what hazardous and non-hazardous lab scenes look like.\n\n"
            "Return the requested format and end with exactly one final label: "
            "hazardous or non-hazardous."
        ),
    })

    return [
        {"role": "system", "content": LAB_SAFETY_GEN_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def answer(
    query: str,
    top_k: int = TOP_K,
    gated_rag: float = GATED_RAG,
) -> dict[str, Any]:
    from retrieval_gating import gate_retrieval_results
    from retriever import hybrid_search

    top_k_retrieved = hybrid_search(query, top_k)
    retrieved = gate_retrieval_results(top_k_retrieved, gated_rag)
    return {
        "query": query,
        "top_k": top_k,
        "gated_rag": float(gated_rag),
        "retrieved_count_before_gate": len(top_k_retrieved),
        "retrieved_count": len(retrieved),
        "retrieved": retrieved,
        "prompt": build_prompt(query, retrieved),
    }
