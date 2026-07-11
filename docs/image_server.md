# Image inference server

`image_server.py` is the only image-serving entry point. The server accepts raw
image bytes and chooses the inference strategy for each request from the
required `mode` query parameter. Backend mode is not selected at startup.

## Modes

| `mode` | Behavior |
| --- | --- |
| `accuracy` | Image RAG followed by VLM inference. Retrieval always uses the `unified_safety` index and a cosine-similarity gate of `0.7`. |
| `latency` | Two-stage VLM-only inference. A short safe/unsafe pass runs first; the annotation pass runs only when the first pass returns unsafe. |

Both modes use the same safety task prompt defined as `SAFETY_PROMPT` in
`image_server.py`. Server inference does not use the default task prompts from
`config.py`.

## Unified accuracy index

The accuracy index must combine all retrieval records from:

- ConstructionSite-10K
- InspecSafe
- LabSafety-Gen

Rebuild the two Chroma collections under:

```text
chroma_db/unified_safety/
```

The directory must contain the existing `siglip2_caption_rag` and
`siglip2_image_rag` collections. IDs must be unique across the three source
datasets; prefixing each ID with its source dataset is recommended. The server
reports the unified image count at startup and returns an inference error for
`mode=accuracy` if this index is unavailable. `mode=latency` does not query the
index.

## Start the server

```bash
python image_server.py --host 0.0.0.0 --port 8000
```

By default, the server preloads SigLIP2 and the VLM so either request mode is
ready. Wait for `Model loading complete. Server is ready.` before sending an
image. The service uses one worker and accepts one inference at a time.

The remaining useful startup options are:

```text
--top-k N
--max-new-tokens N
--stage-one-max-new-tokens N
--stage-two-max-new-tokens N
--max-upload-mb N
--lora-weights PATH
--no-preload
```

`--top-k` is the default number of candidates retrieved in accuracy mode. It
does not affect latency mode. There are no `--rag`, `--dataset`, or
`--gated-rag` startup options.

## Send an image

Accuracy mode:

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://SERVER_IP:8000/infer?mode=accuracy"
```

Latency mode:

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://SERVER_IP:8000/infer?mode=latency"
```

Accuracy mode optionally accepts a per-request `top_k` override:

```bash
curl --data-binary @query.jpg \
  "http://SERVER_IP:8000/infer?mode=accuracy&top_k=8"
```

`mode` is required and accepts only `accuracy` or `latency`. Dataset and gate
query parameters are no longer part of the API.

## Responses

Accuracy response:

```json
{
  "status": "success",
  "mode": "accuracy",
  "response": "<model output>",
  "response_time_seconds": 2.731,
  "rag_dataset": "unified_safety",
  "gated_rag": 0.7,
  "retrieved_count_before_gate": 5,
  "retrieved_count": 2
}
```

Latency response:

```json
{
  "status": "success",
  "mode": "latency",
  "response": "safe",
  "response_time_seconds": 0.842
}
```

An unsafe latency result is returned as `unsafe <short annotation>` when the
second stage supplies an annotation.

While the model is occupied, another request returns HTTP 200 immediately:

```json
{
  "status": "BUSY",
  "mode": "accuracy",
  "response": "",
  "response_time_seconds": ""
}
```

Clients should retry a `BUSY` response after a delay. Busy responses are not
forwarded.

## Health and response forwarding

`GET /health` reports the supported modes, accuracy index name, fixed gate, and
active LoRA weights.

Successful outputs are still forwarded when `RESPONSE_FORWARD_URL` is set in
`config.py`. Forwarding happens after the client response and is excluded from
`response_time_seconds`. Responses include `X-Inference-Mode`,
`X-Inference-Seconds`, and `X-Response-Forwarding` headers.
