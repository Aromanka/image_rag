"""Build a safety reasoning prompt from retrieved historical examples."""

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


def answer(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    retrieved = hybrid_search(query, top_k)
    return {
        "query": query,
        "retrieved": retrieved,
        "prompt": build_prompt(query, retrieved),
    }

