"""
Experiment C: GBM LambdaRank (LGBMRanker, lambdarank/NDCG objective)
This is the primary model under test. Trained with:
  - Monotonic constraints: trust_score=-1, skill_depth=+1
  - LambdaRank (listwise, optimizes NDCG directly)
  - 3-seed stability check
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.evaluation.ablation import run_ablation
from ranking_lab.evaluation.stability_check import run_stability_check
from ranking_lab.evaluation.adversarial_stress_test import run_adversarial_stress_test
from ranking_lab.experiments.common import (
    load_feature_store, load_labels,
    build_training_matrix, build_eval_matrix,
)


def run(feature_store: dict, gold_judgments: dict) -> dict:
    print("\n=== Experiment C: GBM LambdaRank ===")
    labels = load_labels()

    X_train, y_train, train_ids = build_training_matrix(feature_store, labels)
    print(f"  Training on {len(train_ids)} labeled candidates.")

    model = GBMLambdaRankModel(random_state=42)
    model.fit(X_train, y_train)
    model.save("ranking_lab/models/gbm_lambdarank.txt")

    # Eval on gold-set candidates
    gold_cids = list(gold_judgments.keys())
    X_eval, eval_ids = build_eval_matrix(feature_store, gold_cids)

    preds = model.predict(X_eval)
    ranked_ids = [eval_ids[i] for i in np.argsort(-preds)]
    metrics = evaluate_ranking(ranked_ids, gold_judgments)

    print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  NDCG@50: {metrics['ndcg@50']:.4f}")

    # Full corpus eval matrix for ablation
    X_full, full_ids = build_eval_matrix(feature_store, list(feature_store.keys()))

    # Ablation
    print("\n  --- Ablation ---")
    ablation = run_ablation(model, X_full, full_ids, gold_judgments)

    # Stability check (3 seeds)
    print("\n  --- Stability Check (3 seeds) ---")
    stability = run_stability_check(X_train, y_train, X_eval, eval_ids, gold_judgments)

    # Adversarial
    print("\n  --- Adversarial Stress Test ---")
    adversarial = run_adversarial_stress_test(model)

    return {
        "experiment": "C_gbm_lambdarank",
        "model": "GBMLambdaRank",
        **metrics,
        "ablation": ablation,
        "stability": stability,
        "adversarial": adversarial,
    }


if __name__ == "__main__":
    fs = load_feature_store()
    gold = load_gold_set()
    result = run(fs, gold)
    print(f"\nFinal NDCG@10: {result['ndcg@10']:.4f}")
    print(f"Stability: {result['stability']['verdict']}")
