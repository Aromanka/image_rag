"""Build safety reasoning prompts from retrieved historical examples."""

from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, TOP_K
from retriever import hybrid_search


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


def answer(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    retrieved = hybrid_search(query, top_k)
    return {
        "query": query,
        "retrieved": retrieved,
        "prompt": build_prompt(query, retrieved),
    }
