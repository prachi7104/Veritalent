"""
ranking_lab/evaluation/ablation.py

Feature group ablation: evaluates NDCG@10 delta when each feature group
is zeroed out from the trained model's input vector.

Feature groups tested:
  - skill (skill_depth, skill_breadth, skill_recency, skill_mastery_triangulation)
  - career (tenure_stability)
  - activity (activity_quality_composite)
  - trust (trust_score) 
  - logistics (logistics_fit_score)
  - company (product_vs_services)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

FEATURE_GROUPS = {
    "skill":     ["skill_depth", "skill_breadth", "skill_recency", "skill_mastery_triangulation"],
    "career":    ["tenure_stability"],
    "activity":  ["activity_quality_composite"],
    "trust":     ["trust_score"],
    "logistics": ["logistics_fit_score"],
    "company":   ["product_vs_services"],
}


def run_ablation(model, X_all: np.ndarray, candidate_ids: list[str], gold_judgments: dict) -> dict:
    """
    Runs feature group ablation.
    
    For each group, zeros out that group's feature columns in X_all, re-predicts,
    and measures NDCG@10 delta vs. the baseline (all features intact).
    
    Returns dict: {group_name: {"ndcg@10": float, "ndcg@10_delta": float}}
    """
    # Baseline scores
    baseline_preds = model.predict(X_all)
    baseline_order = [candidate_ids[i] for i in np.argsort(-baseline_preds)]
    baseline_metrics = evaluate_ranking(baseline_order, gold_judgments)
    baseline_ndcg = baseline_metrics["ndcg@10"]

    print(f"Baseline NDCG@10: {baseline_ndcg:.4f}")
    results = {"_baseline": baseline_metrics}

    for group_name, features in FEATURE_GROUPS.items():
        indices = [TRAINING_FEATURES.index(f) for f in features if f in TRAINING_FEATURES]
        if not indices:
            continue

        X_ablated = X_all.copy()
        for idx in indices:
            X_ablated[:, idx] = 0.0

        ablated_preds = model.predict(X_ablated)
        ablated_order = [candidate_ids[i] for i in np.argsort(-ablated_preds)]
        ablated_metrics = evaluate_ranking(ablated_order, gold_judgments)
        ablated_ndcg = ablated_metrics["ndcg@10"]

        delta = ablated_ndcg - baseline_ndcg
        print(f"  [{group_name}] NDCG@10={ablated_ndcg:.4f}  delta={delta:+.4f}")
        results[group_name] = {**ablated_metrics, "ndcg@10_delta": delta}

    return results
