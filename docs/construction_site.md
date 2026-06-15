1. preprocess dataset & build rag database from train
```bash
python build_index.py --constructionsite-json data/constructionsite/train.json
```
2. evaluate
```bash
python evaluate_constructionsite10k.py --dataset-json data/constructionsite/test.json --mode rag
python evaluate_constructionsite10k.py --dataset-json data/constructionsite/test.json --mode baseline
```
3. check details
```bash
python utils/evaluate_rag_details.py /root/autodl-tmp/code/image_rag/save/eval_results_constructionsite10k_rag_1781160915.json --demo-dir demo/constructionsite_rag_details --sample-ids 0000001 0000002 0000005 0000007 0000009 0000010
```