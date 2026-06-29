# scripts/validate_shap_fix.py
"""
Verify SHAP values are now unique across candidates.
All-identical SHAP = still broken.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

model = GBMLambdaRankModel()
model.load("ranking_lab/models/gbm_lambdarank.txt")

store = {}
with open("feature_lab/store/feature_store.jsonl") as f:
    for line in f:
        row = json.loads(line)
        store[row["candidate_id"]] = row

sub = pd.read_csv("submission/submission.csv")

import shap
explainer = shap.TreeExplainer(model.model)

print("=== SHAP Top Driver Per Rank (must be DIFFERENT per candidate) ===")
shap_top_features = []
num_feats = model.model.num_feature()
feats_10 = TRAINING_FEATURES[:num_feats]

for rank in [1, 5, 10, 25, 50]:
    row = sub[sub["rank"] == rank].iloc[0]
    cid = row["candidate_id"]
    feats = store.get(cid, {})
    X = np.array([float(feats.get(f, 0) or 0) for f in feats_10])
    shap_vals = explainer.shap_values(X.reshape(1, -1))[0]
    top_idx = np.argmax(np.abs(shap_vals))
    top_feat = feats_10[top_idx]
    top_val  = shap_vals[top_idx]
    shap_top_features.append(top_feat)
    print(f"  Rank {rank:3d}: top driver = {top_feat} ({top_val:+.4f})")

# Check uniqueness — if all identical, still broken
unique_tops = len(set(shap_top_features))
print(f"\nUnique top SHAP drivers across 5 ranks: {unique_tops}/5")
if unique_tops <= 1:
    print("✗ FAIL: Still identical — check that TRAINING_FEATURES (10 features) is used")
elif unique_tops >= 3:
    print("✓ PASS: SHAP values are now candidate-specific")
else:
    print("⚠ PARTIAL: Some differentiation but still fairly uniform")
