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

Test prompt construction before running Qwen2.5-VL:

```bash
python - <<'PY'
import os
from config import DEFAULT_SAFETY_QUERY
from rag_answer import build_image_rag_prompt
from retriever import search_by_query_image

retrieved = search_by_query_image(os.environ['QUERY_IMAGE'], top_k=3)
prompt = build_image_rag_prompt(DEFAULT_SAFETY_QUERY, retrieved)
print(prompt)
PY
```

## 7. Test VLM inference from Python

Baseline inference uses only the query image:

```bash
python - <<'PY'
import os
from vlm_inference import VLM_inference

result = VLM_inference('safety judgement', os.environ['QUERY_IMAGE'])
print(result['output'])
PY
```

RAG inference retrieves similar images and captions, builds the RAG prompt, and
then calls Qwen2.5-VL:

```bash
python - <<'PY'
import os
from vlm_inference import VLM_inference_with_RAG

result = VLM_inference_with_RAG('safety judgement', os.environ['QUERY_IMAGE'], top_k=5)
print('Retrieved IDs:', [item['id'] for item in result['retrieved']])
print(result['output'])
PY
```

You can also run the same workflows through the CLI:

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
