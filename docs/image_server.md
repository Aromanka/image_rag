# Image inference server

`image_server.py` accepts raw image bytes and selects an inference strategy for
each request through the required `mode` query parameter. The mode is selected
per request; it is not fixed when the server starts.

## Modes

| Mode | Query value | Current behavior |
| --- | --- | --- |
| Accuracy-first | `accuracy` | Fine-tuned InspecSafe safety-level inference with image RAG and a fixed `0.7` similarity gate. |
| Latency-first | `latency` | Existing two-stage VLM-only inference. A short safe/unsafe pass runs first, and annotation runs only when needed. |
| Energy-first | `energy` | Placeholder interface; currently uses the same implementation as Accuracy-first. |
| Balanced | `balanced` | Placeholder interface; currently uses the same implementation as Accuracy-first. |

The aliases `accuracy-first`, `latency-first`, `energy-first`, and
`balanced-mode` are also accepted. Responses always report the normalized
values `accuracy`, `latency`, `energy`, or `balanced`.

## Accuracy-first pipeline

Accuracy-first combines the configured fine-tuned model with gated InspecSafe
image RAG:

1. The query image is searched against the `inspecsafe` image index.
2. The top-k results are filtered at cosine similarity `>= 0.7`.
3. The remaining reference images and the query image are passed to the
   configured VLM and LoRA adapter.
4. The model returns the structured InspecSafe assessment used by
   `evaluate_finetuned_inspecsafe.py`:

```json
{
  "scene_description": "<detailed scene description>",
  "hazards": ["<canonical hazard phrase>"],
  "overall_safety_level": "<Level I | Level II | Level III | Level IV>"
}
```

The system prompt and JSON schema come from
`INSPECSAFE_SAFETY_LEVEL_SYSTEM_PROMPT` in `rag_answer.py`. That prompt is kept
identical to `SYSTEM_PROMPT` in `evaluate_finetuned_inspecsafe.py`. The user
instruction is:

```text
Inspect this industrial site image and provide your safety assessment.
```

The base model, processor, and optional LoRA adapter are loaded from
`VLM_MODEL_PATH`, `VLM_PROCESSOR_PATH`, and `VLM_LORA_WEIGHTS` in `config.py`,
using the same model-loading path as the rest of the server. The default output
limit for these structured results is
`INSPECSAFE_SAFETY_LEVEL_MAX_NEW_TOKENS` (currently 384).

Energy-first and Balanced currently execute this exact same pipeline. They are
separate API modes so their implementations can be added later without another
interface change.

## InspecSafe RAG index

Accuracy-first, Energy-first, and Balanced require the two Chroma collections
under:

```text
chroma_db/inspecsafe/
```

The directory contains the existing `siglip2_caption_rag` and
`siglip2_image_rag` collections. The server reports the InspecSafe image count
at startup. Latency-first does not query this index.

Build or rebuild it from the pipeline training data with:

```bash
python build_index.py \
  --dataset-input data/inspecsafe_pipeline/pipeline_train.json \
  --input-format inspecsafe_pipeline \
  --data-root /root/autodl-tmp/data/inspecsafe/DATA_PATH
```

## Start the server

```bash
python image_server.py --host 0.0.0.0 --port 8000
```

By default, the server preloads SigLIP2 and the configured VLM. Wait for
`Model loading complete. Server is ready.` before sending an image. The service
uses one worker and accepts one inference at a time.

Available startup options include:

```text
--top-k N
--max-new-tokens N
--stage-one-max-new-tokens N
--stage-two-max-new-tokens N
--max-upload-mb N
--lora-weights PATH
--no-preload
```

`--top-k` and `--max-new-tokens` affect Accuracy-first, Energy-first, and
Balanced. The two stage token limits affect only Latency-first. A per-request
`top_k` query parameter overrides the startup default for a RAG mode.

## Send an image

Accuracy-first:

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://SERVER_IP:8000/infer?mode=accuracy"
```

Latency-first:

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://SERVER_IP:8000/infer?mode=latency"
```

Energy-first and Balanced placeholders:

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://SERVER_IP:8000/infer?mode=energy"

curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://SERVER_IP:8000/infer?mode=balanced"
```

Per-request retrieval count override:

```bash
curl --data-binary @query.jpg \
  "http://SERVER_IP:8000/infer?mode=accuracy&top_k=8"
```

## Responses

Every successful mode returns the same two semantic fields:

- `safe`: the string `safe` or `unsafe`.
- `annotation`: the normalized scene annotation, without a leading safety
  label.

The original model text remains available in `response` for debugging and
backward compatibility. Each mode has a separate parser:

| Mode | `safe` parser | `annotation` parser |
| --- | --- | --- |
| Accuracy-first | `safe` only when the parsed `hazards` list is empty; otherwise `unsafe`. | Parsed `scene_description`. |
| Latency-first | `safe` only when the first response word is `safe` (case-insensitive); otherwise `unsafe`. | Remove one leading `safe` or `unsafe` word when present; preserve all remaining text. |
| Energy-first | Currently the same semantics as Accuracy-first, through its own parser entry. | Parsed `scene_description`. |
| Balanced | Currently the same semantics as Accuracy-first, through its own parser entry. | Parsed `scene_description`. |

An invalid Accuracy-style JSON response or a non-list/missing `hazards` value
is treated conservatively as `unsafe`; its `annotation` is empty unless a
`scene_description` was parsed.

Accuracy-first response:

```json
{
  "status": "success",
  "mode": "accuracy",
  "safe": "safe",
  "annotation": "No hazards are visible in the inspection scene.",
  "response": "{\"scene_description\":\"No hazards are visible in the inspection scene.\",\"hazards\":[],\"overall_safety_level\":\"Level IV\"}",
  "response_time_seconds": 2.731,
  "rag_dataset": "inspecsafe",
  "gated_rag": 0.7,
  "retrieved_count_before_gate": 5,
  "retrieved_count": 2
}
```

Energy-first and Balanced return the same fields with `mode` set to `energy`
or `balanced`.

Latency-first response:

```json
{
  "status": "success",
  "mode": "latency",
  "safe": "unsafe",
  "annotation": "Smoke is visible near the equipment.",
  "response": "unsafe Smoke is visible near the equipment.",
  "response_time_seconds": 0.842
}
```

For example, the raw latency output `unsafe unsafe Smoke is visible.` produces
`safe: "unsafe"` and `annotation: "unsafe Smoke is visible."`; only the first
label word is removed.

While the model is occupied, another request returns HTTP 200 immediately:

```json
{
  "status": "BUSY",
  "mode": "accuracy",
  "safe": "",
  "annotation": "",
  "response": "",
  "response_time_seconds": ""
}
```

Clients should retry a `BUSY` response after a delay. Busy responses are not
forwarded.

## Health and response forwarding

`GET /health` reports all four normalized modes, the InspecSafe accuracy index,
the fixed gate, the placeholder modes, and active LoRA weights.

Successful outputs are forwarded when `RESPONSE_FORWARD_URL` is set in
`config.py`. Forwarding happens after the client response and is excluded from
`response_time_seconds`. Responses include `X-Inference-Mode`,
`X-Inference-Seconds`, and `X-Response-Forwarding` headers.
