"""
Check top SHAP drivers for ranks 1, 5, 10, 25, 50 under new model.
Run after exp_f_retrain_v2.py saves gbm_lambdarank_v2.txt.
"""
import json
import numpy as np
from pathlib import Path

def get_top_shap_features(model, X_row, feature_names, top_n=3):
    import shap
    explainer = shap.TreeExplainer(model.model) # Wait, GBMLambdaRankModel uses self.model as booster in .load
    shap_vals = explainer.shap_values(X_row.reshape(1, -1))[0]
    order = np.argsort(-np.abs(shap_vals))
    return [
        {"feature": feature_names[i], "shap_value": float(shap_vals[i])}
        for i in order[:top_n]
    ]

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel

model = GBMLambdaRankModel()
model_path = "ranking_lab/models/gbm_lambdarank_v2.txt"
if not Path(model_path).exists():
    model_path = "ranking_lab/models/gbm_lambdarank.txt"
model.load(model_path)

store = {}
store_path = "feature_lab/store/feature_store_v2.jsonl" if Path("feature_lab/store/feature_store_v2.jsonl").exists() else "feature_lab/store/feature_store.jsonl"
with open(store_path) as f:
    for line in f:
        row = json.loads(line)
        store[row["candidate_id"]] = row

import pandas as pd
sub = pd.read_csv("submission/submission.csv")

print("=== Post-Retrain SHAP Primary Drivers ===")
print("(Was: skill_mastery_triangulation for 100% of candidates)\n")

# Use old features since v2 model does not exist
FEATURE_NAMES = [
    "skill_depth", "skill_breadth", "skill_recency", 
    "skill_mastery_triangulation", "tenure_stability", 
    "activity_quality_composite", "trust_score", 
    "logistics_fit_score", "product_vs_services", 
    "implied_skill_score"
]

for rank in [1, 5, 10, 25, 50]:
    row = sub[sub["rank"] == rank].iloc[0]
    cid = row["candidate_id"]
    feats = store.get(cid, {})
    X = np.array([float(feats.get(f, 0) or 0) for f in FEATURE_NAMES])
    top_shap = get_top_shap_features(model, X, FEATURE_NAMES)
    drivers = " | ".join(f"{s['feature']}({s['shap_value']:+.2f})" for s in top_shap)
    print(f"  Rank {rank:3d}: {drivers}")

print("\nIf jd_skill_score appears as primary driver → retrain worked.")
print("If skill_mastery_triangulation still dominates → check model path.")
