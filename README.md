# Image RAG

A construction-safety Image RAG framework using one local
`jinaai/jina-clip-v2` model for both caption and image embeddings.
Model loading is offline-only and never downloads missing model files or code.

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
# Skip this command when dependencies are already installed.
python -m pip install --no-index --find-links /path/to/local/wheelhouse -r requirements.txt
python build_index.py --dataset-csv data/inspecsafe/dataset.csv
uvicorn app:app --reload
```

The installation command only reads packages from a local wheelhouse. Replace
the example path with your local package directory.

The database builder requires `--dataset-csv`, so another dataset can be used
by passing its CSV path to the same command. Each CSV must contain the columns
`id`, `image_path`, `caption`, and `safe_label`.
