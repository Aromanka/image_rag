import os
from config import DEFAULT_SAFETY_QUERY
from rag_answer import build_image_rag_prompt
from retriever import search_by_query_image
from vlm_inference import VLM_inference, VLM_inference_with_RAG


query_image = "/root/autodl-tmp/data/inspecsafe/DATA_PATH/test/Annotations/Normal_data/coal_conveyor-Level04-SuspendedRail-000001/coal_conveyor-Level04-SuspendedRail-000001-001.jpg"
# result = VLM_inference('safety judgement', query_image)
result = VLM_inference_with_RAG('safety judgement', query_image, top_k=3, debug_mode=True)
print(result['output'])
