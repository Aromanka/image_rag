import os
from config import DEFAULT_SAFETY_QUERY
from rag_answer import build_image_rag_prompt
from retriever import search_by_query_image

query_image = "/root/autodl-tmp/data/inspecsafe/DATA_PATH/test/Annotations/Normal_data/coal_conveyor-Level04-SuspendedRail-000001/coal_conveyor-Level04-SuspendedRail-000001-001.jpg"
retrieved = search_by_query_image(query_image, top_k=3)
prompt = build_image_rag_prompt(DEFAULT_SAFETY_QUERY, retrieved)
print(prompt)