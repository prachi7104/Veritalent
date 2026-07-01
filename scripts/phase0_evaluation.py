# scripts/phase0_evaluation.py
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.experiments.common import load_feature_store, build_eval_matrix
from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from submission.blend_scorer import BlendScorer

def evaluate_submission_file(csv_path: str, gold_judgments: dict):
    if not os.path.exists(csv_path):
        print(f"  [Warning] Submission file not found at {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    ranked_ids = df["candidate_id"].tolist()
    metrics = evaluate_ranking(ranked_ids, gold_judgments)
    return metrics

def run():
    print("=== Phase 0: Baseline Evaluation ===")
    
    # 1. Load Gold judgments
    print("Loading gold judgments...")
    gold_judgments = load_gold_set()
    print(f"Loaded {len(gold_judgments)} gold judgments.")
    
    # 2. Load Feature Store
    print("Loading feature store...")
    feature_store = load_feature_store()
    print(f"Loaded {len(feature_store)} candidates from feature store.")
    
    # 3. Evaluate raw GBM model on gold-set candidates
    print("\n--- 3.1 Raw GBM Model (gbm_lambdarank.txt) on Gold Set ---")
    model_path = "ranking_lab/models/gbm_lambdarank.txt"
    if os.path.exists(model_path):
        model = GBMLambdaRankModel()
        model.load(model_path)
        num_feats = model.model.num_feature()
        print(f"  Model loaded. Expects {num_feats} features.")
        
        gold_cids = list(gold_judgments.keys())
        X_eval, eval_ids = build_eval_matrix(feature_store, gold_cids)
        X_eval_sliced = X_eval[:, :num_feats]
        preds = model.model.predict(X_eval_sliced)
        ranked_ids = [eval_ids[i] for i in np.argsort(-preds)]
        
        gbm_metrics = evaluate_ranking(ranked_ids, gold_judgments)
        for k, v in sorted(gbm_metrics.items()):
            print(f"  {k:<10}: {v:.4f}")
    else:
        print(f"  [Error] GBM model not found at {model_path}")
        gbm_metrics = {}

    # 4. Evaluate BlendScorer on gold-set candidates
    print("\n--- 3.2 BlendScorer (Formula: alpha*GBM + (1-alpha)*jd_skill) on Gold Set ---")
    config_path = "ranking_lab/models/blend_config.json"
    if os.path.exists(config_path) and os.path.exists(model_path):
        scorer = BlendScorer(config_path=config_path, model_path=model_path)
        gold_cids = list(gold_judgments.keys())
        
        # BlendScorer.score takes a list of candidate_ids and feature_store
        scores_dict = scorer.score(gold_cids, feature_store)
        # Sort by score descending, then candidate_id ascending for tie-breaks
        sorted_scores = sorted(scores_dict.items(), key=lambda x: (-x[1], x[0]))
        ranked_ids = [x[0] for x in sorted_scores]
        
        blend_metrics = evaluate_ranking(ranked_ids, gold_judgments)
        for k, v in sorted(blend_metrics.items()):
            print(f"  {k:<10}: {v:.4f}")
    else:
        print("  [Error] BlendScorer config or GBM model not found.")
        blend_metrics = {}

    # 5. Evaluate actual submission.csv file
    print("\n--- 3.3 Actual submission.csv File (Top 100 Candidates) ---")
    sub_path = "submission/submission.csv"
    sub_metrics = evaluate_submission_file(sub_path, gold_judgments)
    if sub_metrics:
        for k, v in sorted(sub_metrics.items()):
            print(f"  {k:<10}: {v:.4f}")
    
    # Save baseline report to json
    report = {
        "gbm_model_gold_set": gbm_metrics,
        "blend_scorer_gold_set": blend_metrics,
        "submission_csv": sub_metrics
    }
    
    out_path = Path("ranking_lab/reports/phase0_baseline_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBaseline report saved to {out_path}")

if __name__ == "__main__":
    run()
