# Minimal image inference server

The server accepts image bytes directly, runs inference immediately, prints the
full model output in the terminal, and returns a JSON response. Pure VLM mode is
the default and keeps only the VLM loaded. RAG mode additionally keeps SigLIP2
loaded and uses the selected dataset's vector index.

## Pure VLM mode (without RAG)

Start the server with no mode flag:

```bash
python image_server.py --host 0.0.0.0 --port 8000
```

The request and response interfaces are the same as in RAG mode. `dataset`
still selects the task-specific prompt, but no image retrieval or ChromaDB index
is used. The `top_k` parameter is accepted for interface compatibility and is
ignored in this mode.

## 1. Build one independent index for each dataset

```bash
python build_index.py --dataset-csv data/inspecsafe/train.csv
python build_index.py --constructionsite-json data/constructionsite/train.json
python build_index.py --lab-safety-gen-jsonl data/lab_safety_gen/annotations.jsonl --split train
```

The indexes are stored under `chroma_db/inspecsafe`,
`chroma_db/constructionsite10k`, and `chroma_db/lab_safety_gen`, so they no
longer overwrite each other.

## 2. Start one server process

```bash
python image_server.py --host 0.0.0.0 --port 8000 --rag --dataset inspecsafe --top-k 5 --gated_rag 0.3
python image_server.py --host 0.0.0.0 --port 8000 --rag --dataset constructionsite10k --top-k 3 --gated_rag 0.3
```

Model loading happens during startup. Wait for `Model loading complete. Server is ready.`
before sending the first image.

## 3. Send an image

Use the startup default dataset:

```bash
curl --data-binary @query.jpg -H "Content-Type: image/jpeg" http://SERVER_IP:8000/infer
curl --data-binary @query_image.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:18000/infer?dataset=construction_site"
```

**for autodl server:**
1. terminal 1
```bash
ssh -N \
  -L 18000:127.0.0.1:8000 \
  -p SSH_PORT \
  root@connect.westc.seetacloud.com
```

2. terminal 2(upload image)
```bash
curl --data-binary @query_image.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:18000/infer?dataset=construction_site"
```

Switch the RAG dataset per request without restarting the service:

```bash
curl --data-binary @query.jpg -H "Content-Type: image/jpeg" "http://SERVER_IP:8000/infer?dataset=inspecsafe"
curl --data-binary @query.jpg -H "Content-Type: image/jpeg" "http://SERVER_IP:8000/infer?dataset=construction_site"
curl --data-binary @query.png -H "Content-Type: image/png" "http://SERVER_IP:8000/infer?dataset=lab_safety_gen"
```

Optional `top_k` and `gated_rag` query parameters override the server defaults,
for example `/infer?dataset=inspecsafe&top_k=3&gated_rag=0.3`. The server first
retrieves `top_k`, then removes items whose cosine similarity is below
`gated_rag`. The default threshold is `0`, and zero remaining items is valid.

The response body contains the model output and the complete request latency:

```json
{
  "status": "success",
  "dataset": "inspecsafe",
  "response": "Query image observations: ...",
  "response_time_seconds": 2.731,
  "gated_rag": 0.3,
  "retrieved_count_before_gate": 5,
  "retrieved_count": 2
}
```

The server accepts only one inference request at a time. It does not queue
additional images while the VLM is working. A request received while the VLM
lock is closed returns immediately with HTTP 200 and empty result fields:

```json
{
  "status": "BUSY",
  "dataset": "",
  "response": "",
  "response_time_seconds": ""
}
```

Clients should retry a `BUSY` request after a delay. The busy response is not
forwarded to `RESPONSE_FORWARD_URL`.

`response_time_seconds` measures from the moment an accepted `/infer` request
starts—including reading and validating the image, retrieval, and VLM
generation—until the response is created. Response forwarding runs afterward
and is not included in this value.

Only one GPU inference runs at a time. Extra requests receive `BUSY` instead of
waiting in memory. If the port should not be public, bind to `127.0.0.1` and use
an SSH tunnel instead.

## 4. Forward each model response to another server

On the receiving machine, start the standard-library test receiver:

```bash
python response_receiver.py --host 0.0.0.0 --port 9000
```

To also append every received response to a UTF-8 text file:

```bash
python response_receiver.py --host 0.0.0.0 --port 9000 --output received.txt
```

On the GPU inference server, set the receiver URL in `config.py`:

```python
RESPONSE_FORWARD_URL = "http://RECEIVER_IP:9000/response"
RESPONSE_FORWARD_TIMEOUT_SECONDS = 5.0
```

Restart `image_server.py` after changing the configuration. After each
successful inference, it sends the exact model output as UTF-8 `text/plain`.
The dataset and inference time are included in the `X-Dataset` and
`X-Inference-Seconds` headers. Forwarding runs after the image response has
been returned, and forwarding failures are printed without failing inference.

## 5. Manual test on two servers

Assume:

- GPU inference server: `GPU_SERVER_IP`
- Response receiver server: `RECEIVER_IP`
- Image service port: `8000`
- Response receiver port: `9000`

### Step 1: start the receiver

On the response receiver server:

```bash
cd /path/to/Image_RAG
python response_receiver.py \
  --host 0.0.0.0 \
  --port 9000 \
  --output received_responses.txt
```

The terminal should show:

```text
Listening for model responses on http://0.0.0.0:9000/response
```

Make sure TCP port `9000` is reachable from the GPU server.

### Step 2: test only the receiver

On the GPU server, send a plain-text test message without loading any model:

```bash
curl -i -X POST http://RECEIVER_IP:9000/response \
  -H "Content-Type: text/plain; charset=utf-8" \
  -H "X-Dataset: inspecsafe" \
  --data "test model response"
```

Expected sender response:

```text
HTTP/1.0 204 No Content
```

Expected receiver terminal output contains:

```text
dataset=inspecsafe
test model response
```

The same content should be appended to `received_responses.txt`.

### Step 3: configure forwarding

On the GPU server, edit `config.py`:

```python
RESPONSE_FORWARD_URL = "http://RECEIVER_IP:9000/response"
RESPONSE_FORWARD_TIMEOUT_SECONDS = 5.0
```

Then restart the image inference service:

```bash
python image_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --rag \
  --dataset inspecsafe \
  --top-k 5
```

During startup, verify that it prints:

```text
Response forwarding enabled: http://RECEIVER_IP:9000/response
```

### Step 4: run end-to-end image inference

From any machine that can reach the GPU server:

```bash
curl -i --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://GPU_SERVER_IP:8000/infer?dataset=inspecsafe"
```

The image-service response headers should include:

```text
X-Dataset: inspecsafe
X-Response-Forwarding: scheduled
```

Its JSON response body should include:

```json
{
  "status": "success",
  "dataset": "inspecsafe",
  "response": "<model output>",
  "response_time_seconds": 2.731
}
```

The GPU server should print the inference result followed by a successful
forwarding message:

```text
Response forwarded: url=http://RECEIVER_IP:9000/response status=204
```

The receiver terminal and `received_responses.txt` should contain the exact
model response. Repeat with either of the other RAG datasets if required:

```bash
curl --data-binary @construction.jpg \
  "http://GPU_SERVER_IP:8000/infer?dataset=construction_site"

curl --data-binary @lab.png \
  "http://GPU_SERVER_IP:8000/infer?dataset=lab_safety_gen"
```

If inference succeeds but forwarding fails, verify the receiver IP, port,
firewall rules, and that `/response` is present at the end of
`RESPONSE_FORWARD_URL`.
