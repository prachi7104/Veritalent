# scripts/validate_product_vs_services.py
"""
Validate that the expanded firm list changes the distribution.
Compare before/after by checking how many candidates were reclassified.
"""
import json
import numpy as np

# Load new feature store
scores_new = {}
with open("feature_lab/store/feature_store.jsonl") as f:
    for line in f:
        row = json.loads(line)
        v = row.get("product_vs_services")
        if v is not None:
            scores_new[row["candidate_id"]] = float(v)

arr = np.array(list(scores_new.values()))
print("=== product_vs_services v2 distribution ===")
print(f"  count:          {len(arr)}")
print(f"  mean:           {arr.mean():.4f}")
print(f"  pure product (1.0): {(arr == 1.0).mean():.1%}")
print(f"  pure services (0.0): {(arr == 0.0).mean():.1%}")
print(f"  mixed (0.1-0.9): {((arr > 0.0) & (arr < 1.0)).mean():.1%}")
print(f"  neutral (0.5):  {(arr == 0.5).mean():.1%}")

# Compare baseline if available
import os
from pathlib import Path
baseline_path = sorted(Path("reports_archive").glob("*/feature_distribution_baseline.json"))
if baseline_path:
    with open(baseline_path[-1]) as f:
        baseline = json.load(f)
    old_mean = baseline.get("product_vs_services", {}).get("mean", "N/A")
    print(f"\n=== Comparison vs baseline ===")
    print(f"  old mean: {old_mean}")
    print(f"  new mean: {arr.mean():.4f}")
    if isinstance(old_mean, (int, float)):
        delta = arr.mean() - old_mean
        print(f"  delta:    {delta:+.4f}")
        # SUCCESS CRITERION
        assert abs(delta) >= 0.02, (
            f"FAIL: Distribution changed by only {delta:+.4f}. "
            "Firm list expansion had negligible effect. Check if firm names match."
        )
        print("PASS: Firm list expansion meaningfully changed the distribution.")
