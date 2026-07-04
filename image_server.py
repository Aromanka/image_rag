"""Minimal raw-image HTTP server for low-latency VLM or RAG inference.

Send the image bytes directly in the POST body. No multipart form is required.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import tempfile
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from config import (
    CONSTRUCTIONSITE10K_TASK,
    GATED_RAG,
    LAB_SAFETY_GEN_TASK,
    MAX_TOP_K,
    RESPONSE_FORWARD_TIMEOUT_SECONDS,
    RESPONSE_FORWARD_URL,
    SAFETY_JUDGEMENT_TASK,
    TOP_K,
    VLM_MAX_NEW_TOKENS,
)
from response_forwarding import forward_text_response
from retrieval_gating import validate_gated_rag
from vlm_inference import (
    VLM_inference,
    VLM_inference_with_RAG,
    preload_models,
    preload_vlm_model,
)


DATASET_ALIASES = {
    "inspecsafe": ("inspecsafe", SAFETY_JUDGEMENT_TASK),
    "construction_site": ("constructionsite10k", CONSTRUCTIONSITE10K_TASK),
    "constructionsite": ("constructionsite10k", CONSTRUCTIONSITE10K_TASK),
    "constructionsite10k": ("constructionsite10k", CONSTRUCTIONSITE10K_TASK),
    "lab_safety_gen": ("lab_safety_gen", LAB_SAFETY_GEN_TASK),
    "labsafety_gen": ("lab_safety_gen", LAB_SAFETY_GEN_TASK),
}

IMAGE_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tiff",
}


def _resolve_dataset(value: str) -> tuple[str, str]:
    normalized = value.strip().lower().replace("-", "_")
    resolved = DATASET_ALIASES.get(normalized)
    if resolved is None:
        choices = "inspecsafe, construction_site, lab_safety_gen"
        raise ValueError(f"Unsupported dataset '{value}'. Choose one of: {choices}.")
    return resolved


def _image_suffix(payload: bytes) -> str:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
            image_format = image.format or ""
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Request body is not a valid image.") from exc
    return IMAGE_SUFFIXES.get(image_format.upper(), ".img")


def create_app(
    *,
    default_dataset: str = "inspecsafe",
    default_top_k: int = TOP_K,
    default_gated_rag: float = GATED_RAG,
    max_new_tokens: int = VLM_MAX_NEW_TOKENS,
    max_upload_mb: int = 20,
    preload: bool = True,
    use_rag: bool = False,
) -> FastAPI:
    canonical_default, _ = _resolve_dataset(default_dataset)
    default_gated_rag = validate_gated_rag(default_gated_rag)
    max_upload_bytes = max_upload_mb * 1024 * 1024
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if use_rag:
            from retriever import index_item_count

            datasets = ("inspecsafe", "constructionsite10k", "lab_safety_gen")
            for dataset in datasets:
                try:
                    count = index_item_count(dataset)
                    print(f"RAG index ready: {dataset} ({count} images)", flush=True)
                except RuntimeError as exc:
                    print(f"RAG index unavailable: {dataset} ({exc})", flush=True)

        if preload:
            if use_rag:
                print("Loading SigLIP2 and VLM models...", flush=True)
                await run_in_threadpool(preload_models)
            else:
                print("Loading VLM model...", flush=True)
                await run_in_threadpool(preload_vlm_model)
            print("Model loading complete. Server is ready.", flush=True)
        if RESPONSE_FORWARD_URL:
            print(f"Response forwarding enabled: {RESPONSE_FORWARD_URL}", flush=True)
        else:
            print("Response forwarding disabled (RESPONSE_FORWARD_URL is empty).", flush=True)
        yield

    app = FastAPI(
        title="Image RAG inference" if use_rag else "Image VLM inference",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "default_dataset": canonical_default}

    @app.post("/infer", response_class=JSONResponse)
    async def infer(
        request: Request,
        background_tasks: BackgroundTasks,
        dataset: str | None = Query(default=None),
        top_k: int | None = Query(default=None, ge=1, le=MAX_TOP_K),
        gated_rag: float | None = Query(default=None),
    ) -> JSONResponse:
        request_started = time.perf_counter()
        selected = dataset or canonical_default
        try:
            canonical_dataset, task_type = _resolve_dataset(selected)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image exceeds the {max_upload_mb} MB upload limit.",
                )

        payload = await request.body()
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="POST an image as the raw request body.",
            )
        if len(payload) > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds the {max_upload_mb} MB upload limit.",
            )

        try:
            suffix = _image_suffix(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        effective_top_k = top_k or default_top_k
        try:
            effective_gated_rag = validate_gated_rag(
                gated_rag if gated_rag is not None else default_gated_rag
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        print(
            f"Inference started: mode={'rag' if use_rag else 'vlm'} "
            f"dataset={canonical_dataset} bytes={len(payload)}"
            + (
                f" top_k={effective_top_k} gated_rag={effective_gated_rag}"
                if use_rag
                else ""
            ),
            flush=True,
        )

        try:
            with tempfile.TemporaryDirectory(prefix="image-rag-") as temp_dir:
                image_path = Path(temp_dir) / f"query{suffix}"
                image_path.write_bytes(payload)
                async with inference_lock:
                    if use_rag:
                        result = await run_in_threadpool(
                            VLM_inference_with_RAG,
                            task_type,
                            image_path,
                            top_k=effective_top_k,
                            gated_rag=effective_gated_rag,
                            max_new_tokens=max_new_tokens,
                        )
                    else:
                        result = await run_in_threadpool(
                            VLM_inference,
                            task_type,
                            image_path,
                            max_new_tokens=max_new_tokens,
                        )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Inference rejected: {exc}", flush=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except RuntimeError as exc:
            print(f"Inference failed: {exc}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )

        elapsed = time.perf_counter() - request_started
        output = str(result["output"])
        print(f"Inference result ({elapsed:.3f}s):\n{output}\n", flush=True)
        if RESPONSE_FORWARD_URL:
            background_tasks.add_task(
                forward_text_response,
                RESPONSE_FORWARD_URL,
                output,
                dataset=canonical_dataset,
                inference_seconds=elapsed,
                timeout_seconds=RESPONSE_FORWARD_TIMEOUT_SECONDS,
            )
        response_payload = {
            "dataset": canonical_dataset,
            "response": output,
            "response_time_seconds": round(elapsed, 3),
        }
        if use_rag:
            response_payload.update({
                "gated_rag": effective_gated_rag,
                "retrieved_count": result.get("retrieved_count", 0),
                "retrieved_count_before_gate": result.get(
                    "retrieved_count_before_gate",
                    0,
                ),
            })
        return JSONResponse(
            response_payload,
            headers={
                "X-Dataset": canonical_dataset,
                "X-Inference-Seconds": f"{elapsed:.3f}",
                "X-Response-Forwarding": (
                    "scheduled" if RESPONSE_FORWARD_URL else "disabled"
                ),
            },
            background=background_tasks,
        )

    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve VLM inference from raw image POST requests."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset",
        default="inspecsafe",
        help="Default dataset: inspecsafe, construction_site, or lab_safety_gen.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gated-rag",
        "--gated_rag",
        dest="gated_rag",
        type=float,
        default=GATED_RAG,
        help="Keep top-k RAG results with cosine similarity >= this threshold.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=VLM_MAX_NEW_TOKENS)
    parser.add_argument("--max-upload-mb", type=int, default=20)
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable retrieval-augmented inference (default: pure VLM inference).",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Load models on the first request instead of at startup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if not 1 <= args.top_k <= MAX_TOP_K:
        raise SystemExit(f"--top-k must be between 1 and {MAX_TOP_K}.")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.max_upload_mb < 1:
        raise SystemExit("--max-upload-mb must be at least 1.")
    try:
        _resolve_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    uvicorn.run(
        create_app(
            default_dataset=args.dataset,
            default_top_k=args.top_k,
            default_gated_rag=args.gated_rag,
            max_new_tokens=args.max_new_tokens,
            max_upload_mb=args.max_upload_mb,
            preload=not args.no_preload,
            use_rag=args.rag,
        ),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
    )
