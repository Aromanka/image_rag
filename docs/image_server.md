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
--local-test
--local-test-token TOKEN
--local-test-history-size N
```

`--top-k` and `--max-new-tokens` affect Accuracy-first, Energy-first, and
Balanced. The two stage token limits affect only Latency-first. A per-request
`top_k` query parameter overrides the startup default for a RAG mode.

## Local display test mode

Local test mode is opt-in and does not change any inference backend or mode.
It adds a WebSocket completion channel at `/local-test/ws`. The local display
computer makes the outbound connection, so that computer does not need a public
IP or an inbound firewall rule.

The experiment flow is:

1. The display client loads a local dataset and shows its first image.
2. A successful `/infer` request finishes on the server.
3. The server sends one `inference.completed` event containing the complete
   HTTP result plus query metadata.
4. The client associates the event with the image currently on screen, switches
   immediately to its preloaded next image, and appends the previous image's
   ground truth and server result to JSONL.

`BUSY` responses and rejected/failed inference requests do not advance the
display. Recent completion events are retained in server memory and replayed
when the same client reconnects. Events are assigned UUIDs and the client
de-duplicates them before writing, so a short disconnect does not normally lose
or duplicate a trial. The history is reset when `image_server.py` restarts.

### Start the server in local test mode

Using a shared token is recommended whenever the WebSocket port is reachable by
another machine. The environment variable avoids putting the token directly in
shell history:

```bash
export IMAGE_RAG_LOCAL_TEST_TOKEN='mde450'

python image_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --local-test
```

The server does not select a local display dataset. The portable batch on the
display computer owns all image and ground-truth data. `GET /health` reports
whether local test mode is enabled, the number of display connections, token
requirement, and replay history capacity without exposing the token.

### Start the local fullscreen display

Install the repository requirements on the local computer. Tkinter must also be
available (it is included in standard Windows and macOS Python distributions;
on some Linux distributions it is a separate OS package).

### Export a portable dataset subset

`utils/export_local_test_data.py` copies selected images and rewrites their
annotations into a self-contained directory that can be transferred to the
display computer. The default output is always `data/local_test_batch` (written
as `data\local_test_batch` on Windows), and both supported datasets are exported
together. There is no need to specify `--output-dir` or `--dataset`. For example,
this exports 20 reproducibly shuffled test samples from each dataset:

```bash
python utils/export_local_test_data.py \
  --split test \
  --shuffle \
  --seed 42 \
  --limit 20 \
  --inspecsafe-image-root /root/autodl-tmp/data/inspecsafe/DATA_PATH \
  --labsafety-image-root data/lab_safety_gen
```

`--limit` applies independently to both datasets. Exact entries can be selected
in a defined order by repeating the dataset-specific ID options:

```bash
python utils/export_local_test_data.py \
  --inspecsafe-id 'test__oil_chemical-Level01-example__frame-001' \
  --inspecsafe-id 'test__oil_chemical-Level04-example__frame-002' \
  --labsafety-id 'ls_bench_0021__02'
```

For InspecSafe pipeline records without an explicit `id`, the selectable ID is
the stem of the stored flattened image path. For LabSafety-Gen it is `image_id`.
The output directory is never overwritten. A failed export is cleaned up before
the final directory is created. Rename or remove an earlier
`data/local_test_batch` before producing a new batch.

```text
data/local_test_batch/
├── manifest.json
├── inspecsafe_safety_level/
│   ├── annotations.json
│   └── images/{split}/...
└── labsafety_gen/
    ├── annotations.jsonl
    └── images/{split}/...
```

`manifest.json` records selection settings, original paths, copied paths, file
sizes, and SHA-256 checksums. The exporter reloads both generated annotation
files before finishing, ensuring they are compatible with the display client.
After copying `data/local_test_batch` to the local computer, use the relevant
batch subdirectory as `--image-root`, for example:

```powershell
python utils/local_test_display.py `
  --annotations data/local_test_batch/labsafety_gen/annotations.jsonl `
  --image-root data/local_test_batch/labsafety_gen
```

For InspecSafe:

```powershell
python utils/local_test_display.py \
  --server "ws://SERVER_IP:8000/local-test/ws" \
  --annotations data/local_test_batch/inspecsafe_safety_level/annotations.json \
  --image-root data/local_test_batch/inspecsafe_safety_level \
  --output save/local_test_inspecsafe.jsonl
```

For LabSafety-Gen:

```powershell
python utils/local_test_display.py `
  --server "ws://SERVER_IP:8000/local-test/ws" `
  --annotations data/local_test_batch/labsafety_gen/annotations.jsonl `
  --image-root data/local_test_batch/labsafety_gen `
  --output save/local_test_labsafety.jsonl
```

The viewer is fullscreen by default. Press `Esc` to exit or `F11` to toggle
fullscreen. The next image is decoded and resized while the current image is
displayed. JSONL writes happen on a separate thread after the screen changes.
Useful client options include:

```text
--split {train,test,all}
--shuffle --seed N
--offset N --limit N
--loop
--windowed
--strict-images
--fsync
```

Missing local image files are skipped with a count by default; `--strict-images`
makes the client fail on the first missing file. `--fsync` requests a disk sync
for every result record, trading some disk activity for stronger durability.

If SSH is the only route to the server, open a local port forward first and use
the default `ws://127.0.0.1:8000/local-test/ws` URL:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
```

For a directly exposed connection, prefer `wss://` behind TLS. A token sent over
plain `ws://` is not encrypted.

### Optional trial association hints

The normal display-driven workflow associates results sequentially and requires
no `/infer` changes. If the query producer knows the displayed sample, it may
send hints for mismatch detection:

```bash
curl --data-binary @query.jpg \
  "http://SERVER_IP:8000/infer?mode=accuracy&local_test_sample_id=SAMPLE_ID"
```

Hints are stored under `server_query` in JSONL. The record's `association` is
`matched_hint`, `sample_id_mismatch`, `dataset_mismatch`, or `sequential`.
Each JSONL record also contains the complete local sample metadata and ground
truth, display timestamps/duration, event sequence and UUID, query SHA-256, and
the full normalized server result.

## Send an image

Accuracy-first:
> If a local port transfer is activated for AutoDL server, than SERVER_IP is just `127.0.0.1`

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
