# Run and Test Guide

The project is configured to load `jinaai/jina-clip-v2` from:

```text
/root/autodl-tmp/model/jina_clip
```

All commands below are intended to be run from the project root on the AutoDL
Linux host.

## 1. Prepare the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place images referenced by `data/dataset.csv` under `data/images/`, or update
the CSV paths to point to your own images.

## 2. Verify the local Jina CLIP v2 model

```bash
test -d /root/autodl-tmp/model/jina_clip
python -c "from embedding import get_embedding_model; m = get_embedding_model(); print(m)"
python -c "from embedding import encode_query; print(len(encode_query('worker wearing a safety helmet')))"
```

The final command should print `1024`, matching `EMBED_TRUNCATE_DIM` in
`config.py`.

## 3. Build or rebuild both indexes

```bash
python build_index.py
```

By default, each build deletes and recreates the two Jina CLIP v2 collections.
Set `RESET_COLLECTIONS_ON_BUILD = False` in `config.py` only when incremental
upserts are desired.

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

The first request includes model loading and is expected to be slower.

```bash
curl -s -o /dev/null -w "caption: %{time_total}s\n" -X POST "http://127.0.0.1:8000/search/caption" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -s -o /dev/null -w "image: %{time_total}s\n" -X POST "http://127.0.0.1:8000/search/image" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
curl -s -o /dev/null -w "hybrid: %{time_total}s\n" -X POST "http://127.0.0.1:8000/search/hybrid" -H "Content-Type: application/json" -d '{"query":"worker without helmet near excavator","top_k":5}'
```

## 7. Optional static checks

```bash
python -m compileall app.py build_index.py config.py embedding.py rag_answer.py retriever.py
```
