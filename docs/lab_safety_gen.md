# LabSafety-v1 Generated Lab Safety Task

Synthetic laboratory-safety image classification dataset under `data/lab_safety_gen`.
The task is binary visual hazard recognition with labels `hazardous` and
`non-hazardous`.

Dataset files:
- `data/lab_safety_gen/annotations.jsonl`: one JSON object per image, including `image_id`, `image`, `split`, `safety_label`, `hazards`, `description`, `vlm_label`, and `agree`.
- `data/lab_safety_gen/labels.csv`: tabular copy of the same labels and annotations.
- `data/lab_safety_gen/summary.json`: dataset counts.
- `data/lab_safety_gen/DATASHEET.md`: provenance and caveats.

Current split summary:
- total: 1092 images
- train: 928 images
- test: 164 images
- labels: 321 `hazardous`, 771 `non-hazardous`

1. Build RAG database from train
```bash
python build_index.py --lab-safety-gen-jsonl data/lab_safety_gen/annotations.jsonl --split train
```

2. Evaluate
```bash
python evaluate_labsafety_gen.py --annotations-jsonl data/lab_safety_gen/annotations.jsonl --split test --mode rag
python evaluate_labsafety_gen.py --annotations-jsonl data/lab_safety_gen/annotations.jsonl --split test --mode baseline
```

Optional quick smoke run:
```bash
python evaluate_labsafety_gen.py --annotations-jsonl data/lab_safety_gen/annotations.jsonl --split test --mode rag --limit 10
python evaluate_labsafety_gen.py --annotations-jsonl data/lab_safety_gen/annotations.jsonl --split test --mode baseline --limit 10
```

3. Check details
```bash
python utils/evaluate_rag_details.py save/eval_results_labsafety_gen_rag_1782311673.json --dataset-type lab_safety_gen --demo-dir demo/lab_safety_gen_rag_details --sample-ids ls_bench_0021__02 ls_bench_0025__01 ls_bench_0044__02 ls_bench_0046__03 ls_bench_0053__03 ls_bench_0054__00
```

4. API task type
```json
{
  "task_type": "lab_safety_gen",
  "query_image": "data/lab_safety_gen/images/test/<image_id>.png",
  "query": "Classify this laboratory scene as hazardous or non-hazardous.",
  "top_k": 5
}
```

Use `/vlm/inference` for baseline inference and `/vlm/rag-inference` for full
retrieval-augmented inference.

Notes:
- The RAG index should be built from the `train` split only.
- The evaluator defaults to the `test` split.
- `safety_label` is the ground truth. `vlm_label` and `agree` are retained as metadata and reference context, but they are not used as the target label.
