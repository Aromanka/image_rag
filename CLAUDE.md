# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image_RAG is a construction-safety visual inspection system. Given a query image from a construction site, it retrieves visually similar historical images from a ChromaDB vector database (using SigLIP2 embeddings), then feeds the query image plus retrieved context into Qwen2.5-VL (3B) to produce a safety/unsafe classification with reasoning.

Dataset: InspecSafe-V1 (construction-safety images with `safe`/`unsafe` labels and captions).

## Commands

```bash
# Install dependencies (offline, from local wheelhouse)
python -m pip install --no-index --find-links /path/to/local/wheelhouse -r requirements.txt

# Build ChromaDB indexes from dataset
python build_index.py --dataset-csv data/InspecSafe/dataset.csv

# Start API server
uvicorn app:app --reload

# CLI inference
python vlm_inference.py /path/to/query.jpg --top-k 5    # RAG mode
python vlm_inference.py /path/to/query.jpg --baseline    # Baseline (no retrieval)

# Static checks (no GPU/models needed)
python -m compileall app.py build_index.py config.py embedding.py rag_answer.py retriever.py vlm_inference.py
```

There is no pytest suite or CI. Testing is manual per `docs/TEST.md`.

## Architecture

```
build_index.py  ──→  embedding.py  ──→  chroma_db/  (persistent vector store)
                         ↑
app.py (FastAPI) ──→  retriever.py  ──→  embedding.py  (query encoding)
       │                    │
       └──→  rag_answer.py  (prompt construction from retrieved examples)
                    │
                    └──→  vlm_inference.py  (Qwen2.5-VL generation)
```

**Key modules:**
- `config.py` — all paths, collection names, model params. Edit this to change model paths or retrieval parameters.
- `embedding.py` — SigLIP2 encoder; provides `encode_query()` (text) and `encode_image()` (image) functions.
- `build_index.py` — one-time CLI to populate ChromaDB collections from the dataset CSV.
- `retriever.py` — retrieval strategies: caption search, image embedding search, query-image search, hybrid (RRF).
- `rag_answer.py` — constructs the multi-image RAG prompt from retrieved results.
- `vlm_inference.py` — loads Qwen2.5-VL, exposes `VLM_inference()` and `VLM_inference_with_RAG()`.
- `app.py` — FastAPI server wrapping all the above as HTTP endpoints.

**ChromaDB collections** (cosine similarity, HNSW):
- `siglip2_caption_rag` — text embeddings of image captions
- `siglip2_image_rag` — SigLIP2 image embeddings

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/search/caption` | Caption-to-caption retrieval |
| POST | `/search/image` | Text-to-image retrieval |
| POST | `/search/query-image` | Image-to-image retrieval |
| POST | `/search/hybrid` | Hybrid (RRF) retrieval |
| POST | `/rag/answer` | Retrieval + prompt construction (no VLM) |
| POST | `/vlm/inference` | Baseline VLM inference |
| POST | `/vlm/rag-inference` | Full RAG + VLM pipeline |

## Environment Requirements

- Python 3.10+
- GPU with sufficient VRAM for Qwen2.5-VL 3B
- Local model snapshots (configured in `config.py`):
  - SigLIP2: `/root/autodl-tmp/model/siglip2`
  - Qwen2.5-VL: `/root/autodl-tmp/model/qwenvl_2_5_3B`
- Target platform is AutoDL Linux host; model loading is offline-only (never downloads)

## Dataset CSV Format

The dataset CSV must contain columns: `id`, `image_path`, `caption`, `safe_label`.
