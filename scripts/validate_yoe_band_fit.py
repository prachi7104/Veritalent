# scripts/validate_yoe_band_fit.py
"""
Validate YOE band fit feature distribution and independence from tenure_stability.
"""
import json
import numpy as np
from scipy import stats

yoe_fit = []
tenure = []

with open("feature_lab/store/feature_store.jsonl") as f:
    for line in f:
        row = json.loads(line)
        y = row.get("yoe_band_fit")
        t = row.get("tenure_stability")
        if y is not None and t is not None:
            yoe_fit.append(float(y))
            tenure.append(float(t))

yf = np.array(yoe_fit)
ts = np.array(tenure)

print("=== yoe_band_fit distribution ===")
print(f"  count: {len(yf)}")
print(f"  mean:  {yf.mean():.4f}")
print(f"  score=1.0 (target band): {(yf == 1.0).mean():.1%}")
print(f"  score>=0.70:             {(yf >= 0.70).mean():.1%}")
print(f"  score<=0.40:             {(yf <= 0.40).mean():.1%}")
print(f"  score=0.50 (unknown):    {(yf == 0.50).mean():.1%}")

corr, pval = stats.pearsonr(yf, ts)
print(f"\n=== Independence check ===")
print(f"  Pearson r with tenure_stability: {corr:.4f} (p={pval:.2e})")

# SUCCESS CRITERIA
target_rate = (yf == 1.0).mean()
assert target_rate >= 0.05, (
    f"FAIL: Only {target_rate:.1%} candidates in target band (5-9 YOE). "
    "Dataset may not have enough mid-senior candidates, or feature is broken."
)
assert abs(corr) < 0.70, (
    f"FAIL: yoe_band_fit too correlated with tenure_stability ({corr:.4f}). "
    "Not providing independent signal."
)
print("\nPASS: yoe_band_fit is valid and independent.")
