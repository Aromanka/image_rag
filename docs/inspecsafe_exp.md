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
python evaluate_inspecsafe.py --mode rag --top-k 5 --limit 1000 --dataset-csv data/inspecsafe/test.csv
python utils/evaluate_rag_details.py /root/autodl-tmp/code/image_rag/save/eval_results_rag_1781179356.json --demo-dir demo/inspecsafe_rag_details --sample-ids 1015 175 1132 61 526 1234
```

InspecSafe RAG: `/root/autodl-tmp/code/image_rag/save/eval_results_rag_1781179356.json`
constructionsite RAG: `/root/autodl-tmp/code/image_rag/save/eval_results_constructionsite10k_rag_1781160915.json`
