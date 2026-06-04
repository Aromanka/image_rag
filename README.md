# Image RAG

A construction-safety Image RAG framework using one local
`jinaai/jina-clip-v2` model for both caption and image embeddings.

Supported retrieval modes:

- Caption-to-caption retrieval
- Text-to-image retrieval
- Hybrid retrieval with reciprocal rank fusion
- Retrieval-grounded safety reasoning prompt generation

The configured model path and embedding options are in `config.py`. All setup,
indexing, run, and test commands are documented in `TEST.md`.

## Run the InspecSafe pipeline

From the project root, install the dependencies, build the indexes from the
InspecSafe CSV, and start the API:

```bash
python -m pip install -r requirements.txt
python build_index.py --dataset-csv data/InspecSafe/dataset.csv
uvicorn app:app --reload
```

The database builder requires `--dataset-csv`, so another dataset can be used
by passing its CSV path to the same command. Each CSV must contain the columns
`id`, `image_path`, `caption`, and `safe_label`.
