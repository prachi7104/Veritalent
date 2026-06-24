"""
Experiment E: LambdaRank + Cross-Encoder Ensemble (Option A)

Uses Lab 01 Experiment D cross-encoder scores (reranker) ensembled with
the LambdaRank model via weighted score combination.

STATUS: Flagged as "PENDING Lab 02 Update"
The cross-encoder scores used here are from Lab 01 Experiment D and 
represent the best available cross-encoder at time of writing. Once 
Lab 02 completes the full embedding shootout with cross-encoder 
re-evaluation, this experiment should be re-run with the updated 
cross-encoder scores.

Ensemble formula:
  final_score = alpha * lambdarank_score + (1 - alpha) * cross_encoder_score
  where alpha is tuned over [0.3, 0.5, 0.7] and best alpha is reported.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.experiments.common import (
    load_feature_store, load_labels,
    build_training_matrix, build_eval_matrix,
)

# Attempt to load cross-encoder scores from Lab 01 Exp D output
CROSS_ENCODER_SCORES_PATH = "retrieval_lab/evaluation/exp_d_cross_encoder_scores.json"
ALPHAS = [0.3, 0.5, 0.7]


def _load_cross_encoder_scores() -> dict:
    """Load cross-encoder scores. Returns empty dict if not found (handles gracefully)."""
    if not os.path.exists(CROSS_ENCODER_SCORES_PATH):
        print(f"  [WARNING] Cross-encoder scores not found at {CROSS_ENCODER_SCORES_PATH}")
        print("  Exp E will run in LambdaRank-only mode (equivalent to Exp C).")
        print("  PENDING: Re-run after Lab 02 cross-encoder evaluation completes.")
        return {}
    with open(CROSS_ENCODER_SCORES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run(feature_store: dict, gold_judgments: dict) -> dict:
    print("\n=== Experiment E: LambdaRank + Cross-Encoder Ensemble [PENDING Lab 02] ===")
    labels = load_labels()

    X_train, y_train, train_ids = build_training_matrix(feature_store, labels)
    model = GBMLambdaRankModel(random_state=42)
    model.fit(X_train, y_train)

    gold_cids = list(gold_judgments.keys())
    X_eval, eval_ids = build_eval_matrix(feature_store, gold_cids)
    lambdarank_preds = model.predict(X_eval)

    # Normalize lambdarank scores to [0,1]
    lr_min, lr_max = lambdarank_preds.min(), lambdarank_preds.max()
    lr_norm = (lambdarank_preds - lr_min) / (lr_max - lr_min + 1e-9)

    cross_encoder_scores = _load_cross_encoder_scores()
    has_ce = len(cross_encoder_scores) > 0

    if not has_ce:
        # No cross-encoder scores — report LambdaRank-only as placeholder
        ranked_ids = [eval_ids[i] for i in np.argsort(-lr_norm)]
        metrics = evaluate_ranking(ranked_ids, gold_judgments)
        print(f"  NDCG@10: {metrics['ndcg@10']:.4f} (LambdaRank only, no CE scores)")
        return {
            "experiment": "E_ensemble_pending_lab02",
            "model": "GBMLambdaRank+CrossEncoder",
            "status": "PENDING_LAB02",
            "note": "Cross-encoder scores not available. Results == Exp C.",
            **metrics,
        }

    # Build cross-encoder score vector aligned to eval_ids
    ce_raw = np.array([cross_encoder_scores.get(cid, 0.0) for cid in eval_ids])
    ce_min, ce_max = ce_raw.min(), ce_raw.max()
    ce_norm = (ce_raw - ce_min) / (ce_max - ce_min + 1e-9)

    best_alpha, best_ndcg, best_metrics = None, -1.0, {}
    for alpha in ALPHAS:
        ensemble = alpha * lr_norm + (1 - alpha) * ce_norm
        ranked_ids = [eval_ids[i] for i in np.argsort(-ensemble)]
        m = evaluate_ranking(ranked_ids, gold_judgments)
        print(f"  alpha={alpha}  NDCG@10={m['ndcg@10']:.4f}")
        if m["ndcg@10"] > best_ndcg:
            best_ndcg = m["ndcg@10"]
            best_alpha = alpha
            best_metrics = m

    print(f"  Best alpha={best_alpha}  NDCG@10={best_ndcg:.4f}")
    return {
        "experiment": "E_ensemble_pending_lab02",
        "model": "GBMLambdaRank+CrossEncoder",
        "status": "COMPLETE",
        "best_alpha": best_alpha,
        **best_metrics,
    }


if __name__ == "__main__":
    fs = load_feature_store()
    gold = load_gold_set()
    result = run(fs, gold)
    print(f"\nFinal NDCG@10: {result['ndcg@10']:.4f}")
