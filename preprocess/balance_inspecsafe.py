import argparse
import pandas as pd

def balance_csv(input_csv, output_csv, ratio=0.5, seed=42):
    df = pd.read_csv(input_csv)

    safe_df = df[df["safe_label"] == "safe"]
    unsafe_df = df[df["safe_label"] == "unsafe"]

    # ratio 表示 safe 在最终数据中的占比，默认 0.5 即 safe:unsafe = 1:1
    n_safe = len(safe_df)
    n_unsafe = len(unsafe_df)

    max_total_by_safe = int(n_safe / ratio)
    max_total_by_unsafe = int(n_unsafe / (1 - ratio))
    total = min(max_total_by_safe, max_total_by_unsafe)

    target_safe = int(total * ratio)
    target_unsafe = total - target_safe

    balanced = pd.concat([
        safe_df.sample(n=target_safe, random_state=seed),
        unsafe_df.sample(n=target_unsafe, random_state=seed)
    ])

    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    balanced.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    balance_csv(args.input_csv, args.output_csv, args.ratio, args.seed)

# python balance_csv.py data/inspecsafe/test.csv data/inspecsafe/test_balanced.csv
