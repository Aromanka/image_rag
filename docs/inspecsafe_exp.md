1. preprocess dataset
```bash
python preprocess/InspecSafe_1.py \
  --data_root /root/autodl-tmp/data/inspecsafe/DATA_PATH \
  --output_csv data/inspecsafe/test.csv \
  --split test
python preprocess/InspecSafe_1.py \
  --data_root /root/autodl-tmp/data/inspecsafe/DATA_PATH \
  --output_csv data/inspecsafe/train.csv \
  --split train
```
2. build rag database from train
```bash
python build_index.py --dataset-csv data/inspecsafe/train.csv
```
3. evaluate
```bash
python evaluate_inspecsafe.py --dataset-csv data/inspecsafe/test.csv
python evaluate_inspecsafe.py --mode baseline --dataset-csv data/inspecsafe/test_balanced.csv
python evaluate_inspecsafe.py --mode two-stage --dataset-csv data/inspecsafe/test_balanced.csv
python evaluate_inspecsafe.py --mode rag --top-k 5 --gated_rag 0.3 --limit 1000 --dataset-csv data/inspecsafe/test.csv
python utils/evaluate_rag_details.py /root/autodl-tmp/code/image_rag/save/eval_results_rag_1781179356.json --demo-dir demo/inspecsafe_rag_details --sample-ids 1015 175 1132 61 526 1234
```

## Safety-level evaluation

The four-level evaluator consumes the same pipeline JSON format as the
fine-tuned-model reference scripts. Each sample contains an `image` field and
an assistant message whose content is a JSON string with
`scene_description`, `hazards`, and `overall_safety_level`.

```bash
# Baseline (query image only)
python evaluate_inspecsafe_safety_level.py \
  --mode baseline \
  --dataset-json /root/autodl-tmp/pipeline_test.json \
  --data-root /root/autodl-tmp/data/inspecsafe/DATA_PATH \
  --limit 10

# Image RAG
python evaluate_inspecsafe_safety_level.py \
  --mode rag \
  --dataset-json /root/autodl-tmp/pipeline_test.json \
  --data-root /root/autodl-tmp/data/inspecsafe/DATA_PATH \
  --top-k 5 \
  --gated_rag 0.3
```

The result JSON reports the same metric set as the reference evaluator:
JSON parse rate; safety-level accuracy; per-level and macro/micro precision,
recall, and F1; hazard micro precision/recall/F1; and scene-description SBERT
similarity. The default SBERT path is `SBERT_MODEL_PATH` from `config.py`.
Pass `--skip-scene-metrics` when the local SBERT snapshot is unavailable.

Pipeline paths such as
`images/test__oil_chemical-Level01-Wheeled-002319__oil_chemical-Level01-Wheeled-002319-001.jpg`
are converted to the original server tree automatically. Level04 is resolved
under `Normal_data`; Levels 01-03 are resolved under `Anomaly_data`. Use the
legacy `--image-root /root/autodl-tmp/pipeline_images` option only when the
flattened image directory is available instead.

Saved outputs can be rescored without rerunning the VLM:

```bash
python utils/evaluate_utils.py save/eval_results_inspecsafe_safety_level_rag_*.json \
  --dataset-type inspecsafe_safety_level \
  --sbert-path /root/autodl-tmp/all-MiniLM-L6-v2
```

The shared `/vlm/inference` and `/vlm/rag-inference` APIs also accept
`"task_type": "safety level"`. Existing `"safety judgement"` requests retain
their binary safe/unsafe behavior.

InspecSafe RAG: `/root/autodl-tmp/code/image_rag/save/eval_results_rag_1781179356.json`
constructionsite RAG: `/root/autodl-tmp/code/image_rag/save/eval_results_constructionsite10k_rag_1781160915.json`

RAG modes accept `--gated_rag` (also spelled `--gated-rag`), with a default of
`0`. Retrieval first selects `top_k`, then removes results whose cosine
similarity (`1 - cosine distance`) is below the threshold. Zero remaining
references is valid.

## Two-stage InspecSafe inference

The `two-stage` evaluation mode uses the same dataset and metrics as the
existing InspecSafe evaluation:

1. Stage one generates at most 8 new tokens and requests only `safe` or
   `unsafe`. A result other than an unambiguous `unsafe` is finalized as
   `safe` without another generation.
2. Stage two runs only after stage one returns `unsafe`. It generates at most
   128 new tokens, provides a short annotation, and makes an independent final
   judgement.
3. The final result is `unsafe` only when both stages return `unsafe`.
   Otherwise it is `safe` and its annotation is empty.

The limits can be overridden for experiments:

```bash
python evaluate_inspecsafe.py \
  --mode two-stage \
  --dataset-csv data/inspecsafe/test.csv \
  --stage-one-max-new-tokens 8 \
  --stage-two-max-new-tokens 128
```

Saved evaluation samples retain `stage_one`, `stage_two`, and `annotation`
fields for prompt/output debugging. Their normalized `output` field remains
`safe` or `unsafe`, so the existing `evaluate_results_json` metric path works
without special parsing.

The same pipeline is available over HTTP:

```bash
curl -X POST "http://127.0.0.1:8000/vlm/two-stage-inference" \
  -H "Content-Type: application/json" \
  -d '{"query_image":"/path/to/query.jpg"}'
```

The response always includes `label`, `annotation`, `stage_one`, and
`stage_two`. `stage_two` is `null` when the first stage does not return
`unsafe`.
