"""
Experiment D: LambdaRank with synthetic labels only (no LLM judging).
Control experiment to quantify the value of LLM-judged labels.
Uses the same model architecture as Exp C but trains on synthetic formula
labels regardless of whether LLM labels are available.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.experiments.common import (
    load_feature_store,
    build_training_matrix, build_eval_matrix,
)

SYNTH_LABELS_PATH = "ranking_lab/labels/synthetic_formula_labels.json"


def run(feature_store: dict, gold_judgments: dict) -> dict:
    print("\n=== Experiment D: LambdaRank (Synthetic Labels Only — Control) ===")

    with open(SYNTH_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    X_train, y_train, train_ids = build_training_matrix(feature_store, labels)
    print(f"  Training on {len(train_ids)} labeled candidates (synthetic only).")

    model = GBMLambdaRankModel(random_state=42)
    model.fit(X_train, y_train)

    gold_cids = list(gold_judgments.keys())
    X_eval, eval_ids = build_eval_matrix(feature_store, gold_cids)

    preds = model.predict(X_eval)
    ranked_ids = [eval_ids[i] for i in np.argsort(-preds)]
    metrics = evaluate_ranking(ranked_ids, gold_judgments)

    print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  NDCG@50: {metrics['ndcg@50']:.4f}")
    print("  [Note] This is the control. Compare NDCG@10 with Exp C to measure LLM label value.")

    return {
        "experiment": "D_lambdarank_synth_labels_control",
        "model": "GBMLambdaRank",
        "label_source": "synthetic_formula",
        **metrics,
    }


if __name__ == "__main__":
    fs = load_feature_store()
    gold = load_gold_set()
    result = run(fs, gold)
    print(f"\nFinal NDCG@10: {result['ndcg@10']:.4f}")
