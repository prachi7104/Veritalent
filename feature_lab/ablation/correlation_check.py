"""
correlation_check.py — Flag highly correlated feature pairs for de-duplication.

Threshold: pairwise Pearson correlation > 0.7.
Any cluster above this threshold should be consolidated into a single
composite feature rather than fed separately into the ranking model.

Usage:
    python -m feature_lab.ablation.correlation_check --store <path>
"""
import json
import argparse
import pandas as pd
import numpy as np

CORRELATION_THRESHOLD = 0.7


def run_correlation_check(feature_store_path: str, threshold: float = CORRELATION_THRESHOLD):
    print("Loading feature store for correlation check...")
    data = []
    with open(feature_store_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude reliability tag columns and metadata
    feature_cols = [c for c in numeric_cols if not c.endswith("_reliability") and c != "candidate_id"]

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if val > threshold:
                high_corr.append((idx, col, float(val)))

    high_corr.sort(key=lambda x: x[2], reverse=True)

    if high_corr:
        print(f"\n--- Highly Correlated Feature Pairs (>{threshold}) ---")
        for f1, f2, val in high_corr:
            print(f"  {f1}  <->  {f2}:  {val:.4f}")
    else:
        print(f"No feature pairs with correlation > {threshold} found.")

    return high_corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature correlation checker")
    parser.add_argument("--store", required=True, help="Path to feature_store.jsonl")
    parser.add_argument("--threshold", type=float, default=CORRELATION_THRESHOLD,
                        help=f"Correlation threshold (default: {CORRELATION_THRESHOLD})")
    args = parser.parse_args()
    run_correlation_check(args.store, args.threshold)
