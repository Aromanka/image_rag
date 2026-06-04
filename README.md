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
