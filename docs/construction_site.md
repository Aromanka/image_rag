1. preprocess dataset & build rag database from train
```bash
python build_index.py --constructionsite-json constructionsite_10k/train.json
```
2. evaluate
```bash
python evaluate_constructionsite10k.py --dataset-json constructionsite_10k/test.json --mode rag --limit 10
```
py utils\evaluate_rag_details.py save\eval_results_constructionsite10k_rag_XXXX.json --sample-ids 0000005 0000007