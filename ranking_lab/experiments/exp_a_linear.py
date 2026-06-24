"""
Experiment A: Linear Baseline
Hand-tuned weighted linear combination from Lab 03 weights.
This is the permanent fallback — kept in the codebase forever.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.linear_baseline import LinearBaselineModel
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.experiments.common import load_feature_store, load_labels


def run(feature_store: dict, gold_judgments: dict) -> dict:
    print("\n=== Experiment A: Linear Baseline ===")
    model = LinearBaselineModel()

    # Score all candidates in the feature store
    candidates_list = [{"candidate_id": cid, **feats} for cid, feats in feature_store.items()]
    scores = model.predict_from_dicts(candidates_list)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    metrics = evaluate_ranking(ranked_ids, gold_judgments)

    print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  NDCG@50: {metrics['ndcg@50']:.4f}")
    print(f"  P@10:    {metrics['p@10']:.4f}")
    return {"experiment": "A_linear_baseline", "model": "LinearBaseline", **metrics}


if __name__ == "__main__":
    fs = load_feature_store()
    gold = load_gold_set()
    result = run(fs, gold)
    print(result)
