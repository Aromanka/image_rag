"""FastAPI entry point for Image RAG."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MAX_TOP_K, TOP_K, VLM_MAX_NEW_TOKENS
from rag_answer import answer
from retriever import (
    hybrid_search,
    save_retrieved_images,
    search_by_caption,
    search_by_image_embedding,
    search_by_query_image,
)
from vlm_inference import VLM_inference, VLM_inference_with_RAG


app = FastAPI(
    title="Construction Safety Image RAG",
    description="Caption, SigLIP2 image, and hybrid retrieval APIs.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=MAX_TOP_K)


class CaptionQueryRequest(QueryRequest):
    test_mode: bool = False


class ImagePathRequest(BaseModel):
    query_image: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=MAX_TOP_K)


class VLMInferenceRequest(ImagePathRequest):
    task_type: str = "safety judgement"
    max_new_tokens: int = Field(default=VLM_MAX_NEW_TOKENS, ge=1)


def execute(operation, request: QueryRequest):
    try:
        return operation(request.query, request.top_k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search/caption")
def caption_search(request: CaptionQueryRequest):
    results = execute(search_by_caption, request)
    if not request.test_mode:
        return results
    try:
        return save_retrieved_images(results)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/image")
def image_search(request: QueryRequest):
    return execute(search_by_image_embedding, request)


@app.post("/search/query-image")
def query_image_search(request: ImagePathRequest):
    try:
        return search_by_query_image(request.query_image, request.top_k)
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search/hybrid")
def hybrid(request: QueryRequest):
    return execute(hybrid_search, request)


@app.post("/rag/answer")
def rag(request: QueryRequest):
    return execute(answer, request)


@app.post("/vlm/inference")
def vlm_inference(request: VLMInferenceRequest):
    try:
        return VLM_inference(
            request.task_type,
            request.query_image,
            max_new_tokens=request.max_new_tokens,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/vlm/rag-inference")
def vlm_rag_inference(request: VLMInferenceRequest):
    try:
        return VLM_inference_with_RAG(
            request.task_type,
            request.query_image,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
