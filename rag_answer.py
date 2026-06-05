"""Build safety reasoning prompts from retrieved historical examples."""

from typing import Any

from config import TOP_K
from retriever import hybrid_search


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


def answer(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    retrieved = hybrid_search(query, top_k)
    return {
        "query": query,
        "retrieved": retrieved,
        "prompt": build_prompt(query, retrieved),
    }
