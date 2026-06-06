# Run and Test Guide

All commands below are intended to run from the project root on the AutoDL
Linux host.

The local model paths are configured in `config.py`:

```text
/root/autodl-tmp/model/siglip2
/root/autodl-tmp/model/qwenvl_2_5_3B
```

## 1. Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links /path/to/local/wheelhouse -r requirements.txt
```

Skip installation when dependencies are already available. Replace
`/path/to/local/wheelhouse` with your local package directory if needed.

## 2. Static checks

These checks do not load the local models.

```bash
python -m compileall app.py build_index.py config.py embedding.py rag_answer.py retriever.py vlm_inference.py
python -c "from vlm_inference import build_baseline_prompt; print(build_baseline_prompt('safety judgement'))"
```

The printed prompt should contain:

```text
Is the following image a safe scenario?
```

## 3. Verify dependencies and model paths

```bash
test -d /root/autodl-tmp/model/siglip2
test -d /root/autodl-tmp/model/qwenvl_2_5_3B
python -c "import chromadb, torch, transformers, qwen_vl_utils; print('deps ok')"
```

Verify SigLIP2 can load and encode a text query:

```bash
python -c "from embedding import resolve_model_path; print(resolve_model_path())"
python -c "from embedding import get_embedding_model, get_embedding_processor; print(type(get_embedding_model()), type(get_embedding_processor()))"
python -c "from embedding import encode_query; print(len(encode_query('worker wearing a safety helmet')))"
```

## 4. Build or rebuild indexes

```bash
python build_index.py --dataset-csv data/InspecSafe/dataset.csv
```

This creates both Chroma collections under `chroma_db/`. Rebuild the indexes
after changing the embedding model or dataset.

## 5. Choose a query image

Use any local image path. To reuse the first image from the dataset CSV:

```bash
export QUERY_IMAGE="$(python - <<'PY'
import pandas as pd
df = pd.read_csv('data/InspecSafe/dataset.csv')
print(df.iloc[0]['image_path'])
PY
)"
echo "$QUERY_IMAGE"
test -f "$QUERY_IMAGE"
```

## 6. Test retrieval-only RAG inputs

Test image-to-image retrieval for the query image:

```bash
python - <<'PY'
import os
from retriever import search_by_query_image

results = search_by_query_image(os.environ['QUERY_IMAGE'], top_k=3)
for item in results:
    print(item['id'], item['safe_label'], item['caption'])
PY
```

Test multi-image message construction before running Qwen2.5-VL:

```bash
python - <<'PY'
import os, json
from config import DEFAULT_SAFETY_QUERY
from rag_answer import build_rag_messages
from retriever import search_by_query_image

query_image = os.environ['QUERY_IMAGE']
retrieved = search_by_query_image(query_image, top_k=3)
messages = build_rag_messages(DEFAULT_SAFETY_QUERY, query_image, retrieved)
print(json.dumps(messages, indent=2, ensure_ascii=False))
PY
```

The output should be a messages list where each retrieved image appears as a
`{"type": "image", "image": "<absolute_path>"}` content block followed by a
text annotation, with the query image last.

## 7. VLM inference

### Inference logic

**Baseline** (`VLM_inference`): passes a single query image + text prompt to
Qwen2.5-VL. No retrieval context.

**RAG** (`VLM_inference_with_RAG`): retrieves top-k similar images via
`search_by_query_image`, then calls `build_rag_messages` to construct a
multi-image messages list. Retrieved images are passed as actual image content
blocks (not text paths) interleaved with text annotations, so Qwen2.5-VL can
see both the reference images and the query image. The messages structure:

```python
[
    {"role": "system", "content": "You are a construction safety..."},
    {"role": "user", "content": [
        {"type": "image", "image": "/path/to/ref1.jpg"},
        {"type": "text", "text": "Reference 1: caption (label: safe)"},
        ...
        {"type": "image", "image": "/path/to/query.jpg"},
        {"type": "text", "text": "Query Image: ... Classify ONLY this query image..."},
    ]},
]
```

This messages list is processed by `_run_vlm_messages` which calls
`process_vision_info` and the processor to handle all images in the list.

### Run from Python

```bash
# RAG mode, first 10 samples
python vlm_inference.py --limit 1 --dataset-csv data/inspecsafe/test.csv

# Baseline mode, samples 20-29
python vlm_inference.py --baseline --limit 1 --dataset-csv data/inspecsafe/test.csv

# Custom dataset
python vlm_inference.py --dataset-csv path/to/dataset.csv --top-k 3
```

### Evaluate Model Ability
1. Evaluate all samples with RAG mode (default)
```bash
python evaluate_inspecsafe.py --dataset-csv data/inspecsafe/test.csv
```
1. Evaluate first 50 samples in baseline mode
```bash
python evaluate_inspecsafe.py --mode baseline --limit 100 --dataset-csv data/inspecsafe/test.csv
```
1. Evaluate samples 100-149 with RAG, top-k=3
```bash
python evaluate_inspecsafe.py --mode rag --top-k 3 --offset 100 --limit 50
```

### Run from CLI

```bash
python vlm_inference.py "$QUERY_IMAGE" --baseline
python vlm_inference.py "$QUERY_IMAGE" --top-k 5
```

## 8. Test API endpoints

Start the API in one terminal:

```bash
uvicorn app:app --reload
```

Run these in a second terminal:

```bash
curl "http://127.0.0.1:8000/health"
curl -X POST "http://127.0.0.1:8000/search/query-image" -H "Content-Type: application/json" -d "{\"query_image\":\"$QUERY_IMAGE\",\"top_k\":5}"
curl -X POST "http://127.0.0.1:8000/vlm/inference" -H "Content-Type: application/json" -d "{\"task_type\":\"safety judgement\",\"query_image\":\"$QUERY_IMAGE\"}"
curl -X POST "http://127.0.0.1:8000/vlm/rag-inference" -H "Content-Type: application/json" -d "{\"task_type\":\"safety judgement\",\"query_image\":\"$QUERY_IMAGE\",\"top_k\":5}"
```

The RAG inference response should include `retrieved`, `prompt`, and `output`.
