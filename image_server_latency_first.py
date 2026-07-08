"""Raw-image HTTP server for latency-first two-stage safety inference.

Send image bytes directly in the POST body. No multipart form is required.
"""

import argparse
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
    INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    MAX_TOP_K,
    RESPONSE_FORWARD_TIMEOUT_SECONDS,
    RESPONSE_FORWARD_URL,
    SAFETY_JUDGEMENT_TASK,
)
from response_forwarding import forward_text_response
from retrieval_gating import validate_gated_rag
from vlm_inference import (
    VLM_inference_two_stage,
    active_lora_weights,
    add_lora_cli_arg,
    configure_lora_weights,
    preload_vlm_model,
)


DATASET_ALIASES = {
    "inspecsafe": ("inspecsafe", SAFETY_JUDGEMENT_TASK),
    "safety_judgement": ("inspecsafe", SAFETY_JUDGEMENT_TASK),
    "safety judgement": ("inspecsafe", SAFETY_JUDGEMENT_TASK),
}

IMAGE_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tiff",
}

# This process serves one VLM request at a time. True means available; False
# means an inference request currently owns the model.
VLM_LOCK_OPEN = True


def _resolve_dataset(value: str) -> tuple[str, str]:
    normalized = value.strip().lower().replace("-", "_")
    resolved = DATASET_ALIASES.get(normalized)
    if resolved is None:
        choices = "inspecsafe"
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


def _format_two_stage_response(result: dict) -> str:
    label = str(result.get("label", "")).strip().lower()
    if label != "unsafe":
        return "safe"

    annotation = str(result.get("annotation", "")).strip()
    if annotation:
        return f"unsafe {annotation}"
    return "unsafe"


def create_app(
    *,
    default_dataset: str = "inspecsafe",
    stage_one_max_new_tokens: int = INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    stage_two_max_new_tokens: int = INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    max_upload_mb: int = 20,
    preload: bool = True,
    lora_weights: str | Path | None = None,
) -> FastAPI:
    if lora_weights is not None:
        configure_lora_weights(lora_weights)

    canonical_default, _ = _resolve_dataset(default_dataset)
    max_upload_bytes = max_upload_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            print("Loading VLM model...", flush=True)
            await run_in_threadpool(preload_vlm_model)
            print("Model loading complete. Server is ready.", flush=True)
        if RESPONSE_FORWARD_URL:
            print(f"Response forwarding enabled: {RESPONSE_FORWARD_URL}", flush=True)
        else:
            print("Response forwarding disabled (RESPONSE_FORWARD_URL is empty).", flush=True)
        if active_lora_weights():
            print(f"LoRA weights enabled: {active_lora_weights()}", flush=True)
        else:
            print("LoRA weights disabled.", flush=True)
        yield

    app = FastAPI(
        title="Latency-first two-stage image safety inference",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str | None]:
        return {
            "status": "ok",
            "default_dataset": canonical_default,
            "lora_weights": active_lora_weights(),
        }

    async def _infer_unlocked(
        request: Request,
        background_tasks: BackgroundTasks,
        dataset: str | None,
        top_k: int | None,
        gated_rag: float | None,
    ) -> JSONResponse:
        request_started = time.perf_counter()
        selected = dataset or canonical_default
        try:
            canonical_dataset, task_type = _resolve_dataset(selected)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        if gated_rag is not None:
            try:
                validate_gated_rag(gated_rag)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
        _ = top_k

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

        print(
            f"Inference started: mode=two-stage dataset={canonical_dataset} "
            f"bytes={len(payload)}",
            flush=True,
        )

        try:
            with tempfile.TemporaryDirectory(prefix="image-rag-") as temp_dir:
                image_path = Path(temp_dir) / f"query{suffix}"
                image_path.write_bytes(payload)
                result = await run_in_threadpool(
                    VLM_inference_two_stage,
                    task_type,
                    image_path,
                    stage_one_max_new_tokens=stage_one_max_new_tokens,
                    stage_two_max_new_tokens=stage_two_max_new_tokens,
                )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Inference rejected: {exc}", flush=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except RuntimeError as exc:
            print(f"Inference failed: {exc}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        elapsed = time.perf_counter() - request_started
        output = _format_two_stage_response(result)
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

        return JSONResponse(
            {
                "status": "success",
                "response": output,
                "response_time_seconds": round(elapsed, 3),
            },
            headers={
                "X-Dataset": canonical_dataset,
                "X-Inference-Seconds": f"{elapsed:.3f}",
                "X-Response-Forwarding": (
                    "scheduled" if RESPONSE_FORWARD_URL else "disabled"
                ),
            },
            background=background_tasks,
        )

    @app.post("/infer", response_class=JSONResponse)
    async def infer(
        request: Request,
        background_tasks: BackgroundTasks,
        dataset: str | None = Query(default=None),
        top_k: int | None = Query(default=None, ge=1, le=MAX_TOP_K),
        gated_rag: float | None = Query(default=None),
    ) -> JSONResponse:
        global VLM_LOCK_OPEN

        if not VLM_LOCK_OPEN:
            return JSONResponse(
                {
                    "status": "BUSY",
                    "response": "",
                    "response_time_seconds": "",
                },
                headers={
                    "X-Dataset": "",
                    "X-Inference-Seconds": "",
                    "X-Response-Forwarding": "disabled",
                    "X-VLM-Status": "BUSY",
                },
            )

        VLM_LOCK_OPEN = False
        try:
            return await _infer_unlocked(
                request,
                background_tasks,
                dataset,
                top_k,
                gated_rag,
            )
        finally:
            VLM_LOCK_OPEN = True

    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve latency-first two-stage safety inference from raw images."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset",
        default="inspecsafe",
        help="Default dataset. Only inspecsafe is supported.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Accepted for image_server.py compatibility and ignored.",
    )
    parser.add_argument(
        "--gated-rag",
        "--gated_rag",
        dest="gated_rag",
        type=float,
        default=None,
        help="Accepted for image_server.py compatibility and ignored.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Accepted for image_server.py compatibility and ignored.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Accepted for image_server.py compatibility and ignored.",
    )
    parser.add_argument(
        "--stage-one-max-new-tokens",
        "--stage_one_max_new_tokens",
        dest="stage_one_max_new_tokens",
        type=int,
        default=INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--stage-two-max-new-tokens",
        "--stage_two_max_new_tokens",
        dest="stage_two_max_new_tokens",
        type=int,
        default=INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    )
    parser.add_argument("--max-upload-mb", type=int, default=20)
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Load the VLM on the first request instead of at startup.",
    )
    add_lora_cli_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.top_k is not None and not 1 <= args.top_k <= MAX_TOP_K:
        raise SystemExit(f"--top-k must be between 1 and {MAX_TOP_K}.")
    if args.gated_rag is not None:
        try:
            validate_gated_rag(args.gated_rag)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.stage_one_max_new_tokens < 1:
        raise SystemExit("--stage-one-max-new-tokens must be at least 1.")
    if args.stage_two_max_new_tokens < 1:
        raise SystemExit("--stage-two-max-new-tokens must be at least 1.")
    if args.max_upload_mb < 1:
        raise SystemExit("--max-upload-mb must be at least 1.")
    try:
        _resolve_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    uvicorn.run(
        create_app(
            default_dataset=args.dataset,
            stage_one_max_new_tokens=args.stage_one_max_new_tokens,
            stage_two_max_new_tokens=args.stage_two_max_new_tokens,
            max_upload_mb=args.max_upload_mb,
            preload=not args.no_preload,
            lora_weights=args.lora_weights,
        ),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
    )
