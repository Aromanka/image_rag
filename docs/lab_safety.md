1. build rag database from train
```bash
python build_index.py --lab-safety-json data/lab_safety/lab_train.json
```

2. evaluate
```bash
python evaluate_labsafety.py --dataset-json data/lab_safety/lab_test.json --mode rag --gated_rag 0.3
python evaluate_labsafety.py --dataset-json data/lab_safety/lab_test.json --mode baseline
```

3. check details
```bash
python utils/evaluate_rag_details.py /root/autodl-tmp/code/image_rag/save/eval_results_labsafety_rag_XXXXXXXXXX.json --dataset-type lab_safety --demo-dir demo/lab_safety_rag_details --sample-ids 000083 000091 000025
```

Lab Safety train split: `data/lab_safety/lab_train.json`
Lab Safety test split: `data/lab_safety/lab_test.json`
