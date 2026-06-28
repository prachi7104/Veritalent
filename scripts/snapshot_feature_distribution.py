# scripts/snapshot_feature_distribution.py
"""
Snapshot feature distributions from current feature_store.jsonl.
Run once before any feature changes.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

FEATURE_COLS = [
    "skill_depth", "skill_breadth", "skill_recency",
    "skill_mastery_triangulation", "tenure_stability",
    "activity_quality_composite", "trust_score",
    "logistics_fit_score", "product_vs_services", "implied_skill_score"
]

store_path = Path("feature_lab/store/feature_store.jsonl")
values = defaultdict(list)

with open(store_path) as f:
    for line in f:
        row = json.loads(line)
        for feat in FEATURE_COLS:
            val = row.get(feat)
            if val is not None:
                values[feat].append(float(val))

report = {}
for feat, vals in values.items():
    arr = np.array(vals)
    zero_rate = (arr == 0).mean()
    report[feat] = {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "zero_rate": round(float(zero_rate), 4),
        "nonzero_count": int((arr != 0).sum()),
    }
    print(f"{feat}: mean={report[feat]['mean']:.3f}, std={report[feat]['std']:.3f}, "
          f"zero_rate={report[feat]['zero_rate']:.1%}")

import datetime
out_path = Path(f"reports_archive/baseline_{datetime.date.today().strftime('%Y%m%d')}"
                "/feature_distribution_baseline.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved to {out_path}")
