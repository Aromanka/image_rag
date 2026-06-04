# Run and Test Guide

The project uses the local SigLIP2 model at:

```text
/root/autodl-tmp/model/siglip2
```

Model loading is strictly offline. Missing model or processor files cause
startup to fail instead of being downloaded.

All commands below are intended to run from the project root on the AutoDL
Linux host.

## 1. Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links /path/to/local/wheelhouse -r requirements.txt
```

Skip installation when dependencies are already available. Otherwise, replace
the example wheelhouse path with a local directory containing all required
packages. Do not use an online package index.

Ensure the image paths referenced by the dataset CSV are available on the
machine building the indexes.

## 2. Verify the local SigLIP2 model

```bash
test -d /root/autodl-tmp/model/siglip2
python -c "from embedding import resolve_model_path; print(resolve_model_path())"
python -c "from embedding import get_embedding_model, get_embedding_processor; print(type(get_embedding_model()), type(get_embedding_processor()))"
python -c "from embedding import encode_query; print(len(encode_query('worker wearing a safety helmet')))"
```

The resolved directory must contain a complete SigLIP2 Transformers snapshot,
including `config.json`, model weights, tokenizer files, and processor files.
Its `config.json` must have `"model_type": "siglip2"`.

If validation fails, inspect the local directory:

```bash
find /root/autodl-tmp/model/siglip2 -maxdepth 3 -type f -print
python -c "import json; print(json.load(open('/root/autodl-tmp/model/siglip2/config.json')).get('model_type'))"
```

## 3. Build or rebuild both indexes

```bash
python build_index.py --dataset-csv data/InspecSafe/dataset.csv
```

By default, each build deletes and recreates the two SigLIP2 collections.
Set `RESET_COLLECTIONS_ON_BUILD = False` in `config.py` only when incremental
upserts are desired. To build from another dataset, pass its CSV path through
`--dataset-csv`.

Do not reuse vectors created by another encoder. The application uses new
SigLIP2 collection names, and the build command must run before querying them.

## 4. Run the API

```bash
uvicorn app:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## 5. Test API endpoints

Run these commands in a second terminal while the API is running.

```bash
curl "http://127.0.0.1:8000/health"
```

```bash
curl -X POST "http://127.0.0.1:8000/search/caption" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -X POST "http://127.0.0.1:8000/search/image" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -X POST "http://127.0.0.1:8000/search/hybrid" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -X POST "http://127.0.0.1:8000/rag/answer" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
```

## 6. Basic retrieval latency check

The first request includes local model loading and is expected to be slower.

```bash
curl -s -o /dev/null -w "caption: %{time_total}s\n" -X POST "http://127.0.0.1:8000/search/caption" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -s -o /dev/null -w "image: %{time_total}s\n" -X POST "http://127.0.0.1:8000/search/image" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -s -o /dev/null -w "hybrid: %{time_total}s\n" -X POST "http://127.0.0.1:8000/search/hybrid" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
```

## 7. Optional static checks

```bash
python -m compileall app.py build_index.py config.py embedding.py rag_answer.py retriever.py
```
