"""Raw-image HTTP server with request-selected inference modes.

Send image bytes directly in the POST body and select the backend with the
required ``mode`` query parameter. No multipart form is required.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from config import (
    INSPECSAFE_DATASET,
    INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS,
    INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    MAX_TOP_K,
    RESPONSE_FORWARD_TIMEOUT_SECONDS,
    RESPONSE_FORWARD_URL,
    SAFETY_JUDGEMENT_TASK,
    SAFETY_LEVEL_TASK,
    TOP_K,
    UNIFIED_SAFETY_DATASET,
    VLM_LORA_WEIGHTS,
)
from response_forwarding import forward_text_response
from utils.evaluate_utils import extract_inspecsafe_safety_level_json
from utils.local_test_channel import LocalTestHub
from utils.local_test_data import SUPPORTED_LOCAL_TEST_DATASETS, normalize_dataset
from vlm_inference import (
    VLM_inference_two_stage,
    VLM_inference_two_stage_with_RAG,
    VLM_inference_with_RAG,
    active_lora_weights,
    add_lora_cli_arg,
    configure_lora_weights,
    preload_models,
)


ACCURACY_MODE = "accuracy"
LATENCY_MODE = "latency"
ENERGY_MODE = "energy"
BALANCED_MODE = "balanced"
SUPPORTED_MODES = {ACCURACY_MODE, LATENCY_MODE, ENERGY_MODE, BALANCED_MODE}
SAFETY_LEVEL_RAG_MODES = {ACCURACY_MODE, ENERGY_MODE}
RAG_MODES = SAFETY_LEVEL_RAG_MODES | {BALANCED_MODE}
MODE_ALIASES = {
    "accuracy-first": ACCURACY_MODE,
    "latency-first": LATENCY_MODE,
    "energy-first": ENERGY_MODE,
    "balanced-mode": BALANCED_MODE,
}
ACCURACY_GATE = 0.7
# Balanced mode keeps latency's two-stage decision policy, but calibrates both
# stages with one shared top-3 retrieval performed before stage one.
BALANCED_TOP_K = 3
BALANCED_GATE = ACCURACY_GATE

# Accuracy-first follows the fine-tuned InspecSafe evaluation task. The full
# system prompt and JSON schema are supplied by the safety-level RAG builder.
ACCURACY_QUERY = "Inspect this industrial site image and provide your safety assessment."

# Keep the existing latency-first prompt and two-stage behavior unchanged.
SAFETY_PROMPT = """You are evaluating the overall safety condition shown in the image.

Classify the image as UNSAFE only when clearly visible evidence shows an unsafe
action, missing required personal protective equipment, or a definite workplace
hazard. Otherwise classify it as SAFE."""

IMAGE_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
    "TIFF": ".tiff",
}

# One worker owns one model. Requests received while inference is active return
# BUSY immediately instead of accumulating in GPU memory.
VLM_LOCK_OPEN = True


def _configure_server_lora(lora_weights: str | Path | None) -> str | Path | None:
    """Select the shared server LoRA before any inference mode loads the VLM.

    All modes use one VLM instance. Falling back to the environment/configured
    adapter here explicitly enables the default LoRA for latency-first/two-stage
    inference instead of relying on ``vlm_inference`` import-time configuration.
    """
    if lora_weights is None:
        selected = os.environ.get("VLM_LORA_WEIGHTS") or VLM_LORA_WEIGHTS
    else:
        selected = lora_weights
    configure_lora_weights(selected)
    return selected


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_local_test_event(
    *,
    payload: bytes,
    content_type: str | None,
    dataset: str | None,
    sample_id: str | None,
    response_payload: dict[str, object],
) -> dict[str, object]:
    """Build the versioned completion event sent to local display clients."""
    return {
        "type": "inference.completed",
        "event_id": str(uuid4()),
        "completed_at": _utc_now_iso(),
        "query": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "content_type": content_type,
            "dataset": dataset,
            "sample_id": sample_id,
        },
        "result": response_payload,
    }


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    normalized = MODE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_MODES:
        choices = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"Unsupported mode '{mode}'. Choose one of: {choices}.")
    return normalized


def _image_suffix(payload: bytes) -> str:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
            image_format = image.format or ""
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Request body is not a valid image.") from exc
    return IMAGE_SUFFIXES.get(image_format.upper(), ".img")


def _format_latency_response(result: dict) -> str:
    if str(result.get("label", "")).strip().lower() != "unsafe":
        return "safe"
    annotation = str(result.get("annotation", "")).strip()
    return f"unsafe {annotation}" if annotation else "unsafe"


def _parse_accuracy_response(output: str) -> dict[str, str]:
    parsed = extract_inspecsafe_safety_level_json(output)
    hazards = parsed.get("hazards") if parsed is not None else None
    scene_description = parsed.get("scene_description") if parsed is not None else ""
    annotation = str(scene_description or "").strip()
    return {
        "safe": "safe" if isinstance(hazards, list) and not hazards else "unsafe",
        "annotation": annotation,
    }


def _parse_latency_response(output: str) -> dict[str, str]:
    stripped = str(output).strip()
    parts = stripped.split(maxsplit=1)
    first_word = parts[0].lower() if parts else ""
    if first_word in {"safe", "unsafe"}:
        annotation = parts[1].strip() if len(parts) > 1 else ""
    else:
        annotation = stripped
    return {
        "safe": "safe" if first_word == "safe" else "unsafe",
        "annotation": annotation,
    }


def _parse_energy_response(output: str) -> dict[str, str]:
    return _parse_accuracy_response(output)


def _parse_balanced_response(output: str) -> dict[str, str]:
    return _parse_latency_response(output)


MODE_RESPONSE_PARSERS = {
    ACCURACY_MODE: _parse_accuracy_response,
    LATENCY_MODE: _parse_latency_response,
    ENERGY_MODE: _parse_energy_response,
    BALANCED_MODE: _parse_balanced_response,
}


def _parse_mode_response(mode: str, output: str) -> dict[str, str]:
    return MODE_RESPONSE_PARSERS[mode](output)


def _build_success_response_payload(
    mode: str,
    output: str,
    result: dict,
    elapsed: float,
) -> dict[str, object]:
    response_payload: dict[str, object] = {
        "status": "success",
        "mode": mode,
        "lora_weights": active_lora_weights(),
        **_parse_mode_response(mode, output),
        "response": output,
        "response_time_seconds": round(elapsed, 3),
    }
    if mode in RAG_MODES:
        response_payload.update(
            {
                "rag_task": result.get("task_type"),
                "rag_dataset": result.get("rag_dataset", INSPECSAFE_DATASET),
                "top_k": result.get("top_k"),
                "gated_rag": result.get("gated_rag", ACCURACY_GATE),
                "retrieved_count_before_gate": result.get(
                    "retrieved_count_before_gate", 0
                ),
                "retrieved_count": result.get("retrieved_count", 0),
            }
        )
    return response_payload


def _run_balanced_inference(
    *,
    image_path: Path,
    rag_dataset: str = INSPECSAFE_DATASET,
    stage_one_max_new_tokens: int,
    stage_two_max_new_tokens: int,
) -> tuple[str, dict]:
    result = VLM_inference_two_stage_with_RAG(
        SAFETY_JUDGEMENT_TASK,
        image_path,
        query=SAFETY_PROMPT,
        top_k=BALANCED_TOP_K,
        gated_rag=BALANCED_GATE,
        rag_dataset=rag_dataset,
        stage_one_max_new_tokens=stage_one_max_new_tokens,
        stage_two_max_new_tokens=stage_two_max_new_tokens,
    )
    return _format_latency_response(result), result


def _run_inference(
    *,
    image_path: Path,
    mode: str,
    top_k: int,
    max_new_tokens: int,
    stage_one_max_new_tokens: int,
    stage_two_max_new_tokens: int,
) -> tuple[str, dict]:
    if mode == BALANCED_MODE:
        return _run_balanced_inference(
            image_path=image_path,
            rag_dataset=INSPECSAFE_DATASET,
            stage_one_max_new_tokens=stage_one_max_new_tokens,
            stage_two_max_new_tokens=stage_two_max_new_tokens,
        )

    if mode in SAFETY_LEVEL_RAG_MODES:
        result = VLM_inference_with_RAG(
            SAFETY_LEVEL_TASK,
            image_path,
            query=ACCURACY_QUERY,
            top_k=top_k,
            gated_rag=ACCURACY_GATE,
            rag_dataset=INSPECSAFE_DATASET,
            max_new_tokens=max_new_tokens,
        )
        return str(result["output"]), result

    result = VLM_inference_two_stage(
        SAFETY_JUDGEMENT_TASK,
        image_path,
        query=SAFETY_PROMPT,
        stage_one_max_new_tokens=stage_one_max_new_tokens,
        stage_two_max_new_tokens=stage_two_max_new_tokens,
    )
    return _format_latency_response(result), result


def create_app(
    *,
    default_top_k: int = TOP_K,
    max_new_tokens: int = INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS,
    stage_one_max_new_tokens: int = INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    stage_two_max_new_tokens: int = INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    max_upload_mb: int = 20,
    preload: bool = True,
    lora_weights: str | Path | None = None,
    local_test_mode: bool = False,
    local_test_dataset: str | None = None,
    local_test_history_size: int = 1024,
) -> FastAPI:
    _configure_server_lora(lora_weights)

    max_upload_bytes = max_upload_mb * 1024 * 1024
    normalized_local_test_dataset = (
        normalize_dataset(local_test_dataset) if local_test_dataset else None
    )
    local_test_hub = LocalTestHub(
        enabled=local_test_mode,
        history_size=local_test_history_size,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        from retriever import index_item_count

        try:
            count = index_item_count(INSPECSAFE_DATASET)
            print(
                f"InspecSafe RAG index ready: {INSPECSAFE_DATASET} "
                f"({count} images)",
                flush=True,
            )
        except RuntimeError as exc:
            print(f"InspecSafe RAG index unavailable: {exc}", flush=True)

        if preload:
            print("Loading SigLIP2 and VLM models...", flush=True)
            await run_in_threadpool(preload_models)
            print("Model loading complete. Server is ready.", flush=True)
        if RESPONSE_FORWARD_URL:
            print(f"Response forwarding enabled: {RESPONSE_FORWARD_URL}", flush=True)
        else:
            print("Response forwarding disabled (RESPONSE_FORWARD_URL is empty).", flush=True)
        if active_lora_weights():
            print(f"LoRA weights enabled: {active_lora_weights()}", flush=True)
        else:
            print("LoRA weights disabled.", flush=True)
        if local_test_hub.enabled:
            print(
                "Local test mode enabled: ws://<server>/local-test/ws "
                f"(dataset={normalized_local_test_dataset or 'client-selected'})",
                flush=True,
            )
        else:
            print("Local test mode disabled.", flush=True)
        yield

    app = FastAPI(
        title="Image safety inference",
        version="3.1.0",
        lifespan=lifespan,
    )
    app.state.local_test_hub = local_test_hub

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "modes": sorted(SUPPORTED_MODES),
            "accuracy_rag_dataset": INSPECSAFE_DATASET,
            "accuracy_gate": ACCURACY_GATE,
            "balanced_rag_dataset": INSPECSAFE_DATASET,
            "balanced_top_k": BALANCED_TOP_K,
            "balanced_gate": BALANCED_GATE,
            "placeholder_modes": sorted({ENERGY_MODE}),
            "lora_weights": active_lora_weights(),
            "local_test": {
                "enabled": local_test_hub.enabled,
                "websocket_path": "/local-test/ws",
                "connected_clients": local_test_hub.connection_count,
                "default_dataset": normalized_local_test_dataset,
                "history_size": local_test_hub.history_size,
            },
        }

    @app.websocket("/local-test/ws")
    async def local_test_websocket(
        websocket: WebSocket,
        after_sequence: int = Query(default=-1, ge=-1),
        server_instance_id: str | None = Query(default=None),
    ) -> None:
        if not local_test_hub.enabled:
            await websocket.close(code=1008, reason="Local test mode is disabled.")
            return

        try:
            await local_test_hub.connect(
                websocket,
                after_sequence=after_sequence,
                client_instance_id=server_instance_id,
            )
            while True:
                # The client need not send application messages. Receiving here
                # detects clean disconnects while protocol ping/pong is handled
                # by the WebSocket implementation.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await local_test_hub.disconnect(websocket)

    async def _infer_unlocked(
        request: Request,
        background_tasks: BackgroundTasks,
        mode: str,
        top_k: int | None,
        local_test_dataset_hint: str | None,
        local_test_sample_id: str | None,
    ) -> JSONResponse:
        request_started = time.perf_counter()
        try:
            selected_mode = _normalize_mode(mode)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        try:
            event_dataset = (
                normalize_dataset(local_test_dataset_hint)
                if local_test_dataset_hint
                else normalized_local_test_dataset
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

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
            effective_top_k = (
                BALANCED_TOP_K
                if selected_mode == BALANCED_MODE
                else top_k or default_top_k
            )
            effective_gate = (
                BALANCED_GATE
                if selected_mode == BALANCED_MODE
                else ACCURACY_GATE
            )
            print(
                f"Inference started: mode={selected_mode} bytes={len(payload)}"
                + (
                    f" rag_dataset={INSPECSAFE_DATASET} "
                    f"top_k={effective_top_k} gate={effective_gate}"
                    if selected_mode in RAG_MODES
                    else ""
                ),
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix="image-rag-") as temp_dir:
                image_path = Path(temp_dir) / f"query{suffix}"
                image_path.write_bytes(payload)
                output, result = await run_in_threadpool(
                    _run_inference,
                    image_path=image_path,
                    mode=selected_mode,
                    top_k=effective_top_k,
                    max_new_tokens=max_new_tokens,
                    stage_one_max_new_tokens=stage_one_max_new_tokens,
                    stage_two_max_new_tokens=stage_two_max_new_tokens,
                )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Inference rejected: {exc}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            print(f"Inference failed: {exc}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        elapsed = time.perf_counter() - request_started
        print(f"Inference result ({elapsed:.3f}s):\n{output}\n", flush=True)
        response_payload = _build_success_response_payload(
            selected_mode,
            output,
            result,
            elapsed,
        )

        # Schedule display notification first so an unrelated HTTP forwarder
        # cannot delay the local image transition.
        if local_test_hub.enabled:
            background_tasks.add_task(
                local_test_hub.publish,
                _build_local_test_event(
                    payload=payload,
                    content_type=request.headers.get("content-type"),
                    dataset=event_dataset,
                    sample_id=local_test_sample_id,
                    response_payload=response_payload,
                ),
            )
        if RESPONSE_FORWARD_URL:
            background_tasks.add_task(
                forward_text_response,
                RESPONSE_FORWARD_URL,
                output,
                dataset=(
                    INSPECSAFE_DATASET
                    if selected_mode in RAG_MODES
                    else UNIFIED_SAFETY_DATASET
                ),
                inference_seconds=elapsed,
                timeout_seconds=RESPONSE_FORWARD_TIMEOUT_SECONDS,
            )

        return JSONResponse(
            response_payload,
            headers={
                "X-Inference-Mode": selected_mode,
                "X-Inference-Seconds": f"{elapsed:.3f}",
                "X-Response-Forwarding": (
                    "scheduled" if RESPONSE_FORWARD_URL else "disabled"
                ),
                "X-Local-Test-Notification": (
                    "scheduled" if local_test_hub.enabled else "disabled"
                ),
            },
            background=background_tasks,
        )

    @app.post("/infer", response_class=JSONResponse)
    async def infer(
        request: Request,
        background_tasks: BackgroundTasks,
        mode: str = Query(
            ...,
            description="Inference mode: accuracy, latency, energy, or balanced.",
        ),
        top_k: int | None = Query(
            default=None,
            ge=1,
            le=MAX_TOP_K,
            description="RAG top-k override; balanced mode always uses top-k 3.",
        ),
        local_test_dataset: str | None = Query(
            default=None,
            description="Optional display dataset association hint.",
        ),
        local_test_sample_id: str | None = Query(
            default=None,
            max_length=256,
            description="Optional display sample ID association hint.",
        ),
    ) -> JSONResponse:
        global VLM_LOCK_OPEN

        if not VLM_LOCK_OPEN:
            return JSONResponse(
                {
                    "status": "BUSY",
                    "mode": mode,
                    "safe": "",
                    "annotation": "",
                    "response": "",
                    "response_time_seconds": "",
                },
                headers={
                    "X-Inference-Mode": mode,
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
                mode,
                top_k,
                local_test_dataset,
                local_test_sample_id,
            )
        finally:
            VLM_LOCK_OPEN = True

    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve request-selected image inference modes."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Default RAG top-k; balanced mode always uses top-k 3.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--stage-one-max-new-tokens",
        type=int,
        default=INSPECSAFE_STAGE_ONE_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--stage-two-max-new-tokens",
        type=int,
        default=INSPECSAFE_STAGE_TWO_MAX_NEW_TOKENS,
    )
    parser.add_argument("--max-upload-mb", type=int, default=20)
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Load models on the first request instead of at startup.",
    )
    parser.add_argument(
        "--local-test",
        action="store_true",
        help="Enable the /local-test/ws inference-completion channel.",
    )
    parser.add_argument(
        "--local-test-dataset",
        choices=sorted(SUPPORTED_LOCAL_TEST_DATASETS),
        default=None,
        help="Default dataset hint included in local-test completion events.",
    )
    parser.add_argument(
        "--local-test-history-size",
        type=int,
        default=1024,
        help="Number of recent completion events retained for reconnect replay.",
    )
    add_lora_cli_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if not 1 <= args.top_k <= MAX_TOP_K:
        raise SystemExit(f"--top-k must be between 1 and {MAX_TOP_K}.")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.stage_one_max_new_tokens < 1:
        raise SystemExit("--stage-one-max-new-tokens must be at least 1.")
    if args.stage_two_max_new_tokens < 1:
        raise SystemExit("--stage-two-max-new-tokens must be at least 1.")
    if args.max_upload_mb < 1:
        raise SystemExit("--max-upload-mb must be at least 1.")
    if args.local_test_history_size < 1:
        raise SystemExit("--local-test-history-size must be at least 1.")

    uvicorn.run(
        create_app(
            default_top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            stage_one_max_new_tokens=args.stage_one_max_new_tokens,
            stage_two_max_new_tokens=args.stage_two_max_new_tokens,
            max_upload_mb=args.max_upload_mb,
            preload=not args.no_preload,
            lora_weights=args.lora_weights,
            local_test_mode=args.local_test,
            local_test_dataset=args.local_test_dataset,
            local_test_history_size=args.local_test_history_size,
        ),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
    )
