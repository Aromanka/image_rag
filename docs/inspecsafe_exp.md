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
python evaluate_inspecsafe.py --mode rag --top-k 3 --limit 10 --dataset-csv data/inspecsafe/test.csv
```
