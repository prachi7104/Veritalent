"""
ranking_lab/evaluation/stability_check.py

Retrains the LambdaRank model across 3 random seeds and measures
Spearman rank correlation across resulting orderings.

If mean Spearman r < 0.85 across seeds, the model is considered unstable
and the report must note this explicitly.
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.evaluation.ndcg_eval import evaluate_ranking

SEEDS = [42, 7, 123]
STABILITY_THRESHOLD = 0.85


def run_stability_check(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    eval_candidate_ids: list[str],
    gold_judgments: dict,
) -> dict:
    """
    Trains the LambdaRank model with SEEDS seeds and computes:
      - Per-seed NDCG@10
      - Pairwise Spearman rank correlations between all seed orderings
      - Mean and min Spearman r
      - Stability verdict (STABLE if mean_r >= 0.85, UNSTABLE otherwise)
    """
    all_scores = []
    all_ndcg = []

    for seed in SEEDS:
        model = GBMLambdaRankModel(random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_eval)
        all_scores.append(preds)
        ranked = [eval_candidate_ids[i] for i in np.argsort(-preds)]
        metrics = evaluate_ranking(ranked, gold_judgments)
        all_ndcg.append(metrics["ndcg@10"])
        print(f"  Seed {seed}: NDCG@10={metrics['ndcg@10']:.4f}")

    # Pairwise Spearman correlations
    correlations = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            r, _ = spearmanr(all_scores[i], all_scores[j])
            correlations.append(r)
            print(f"  Spearman(seed={SEEDS[i]}, seed={SEEDS[j]}): {r:.4f}")

    mean_r = float(np.mean(correlations))
    min_r = float(np.min(correlations))
    verdict = "STABLE" if mean_r >= STABILITY_THRESHOLD else "UNSTABLE"

    print(f"\nStability verdict: {verdict} (mean_r={mean_r:.4f}, threshold={STABILITY_THRESHOLD})")

    return {
        "seeds": SEEDS,
        "ndcg_per_seed": dict(zip(SEEDS, all_ndcg)),
        "mean_ndcg": float(np.mean(all_ndcg)),
        "pairwise_spearman": correlations,
        "mean_spearman_r": mean_r,
        "min_spearman_r": min_r,
        "stability_threshold": STABILITY_THRESHOLD,
        "verdict": verdict,
    }
