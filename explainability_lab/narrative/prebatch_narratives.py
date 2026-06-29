# explainability_lab/narrative/prebatch_narratives.py
"""
Prebatches (pre-computes) and caches the narratives for the top-100 candidates.
Uses the blend-aware scoring strategy (alpha=0.9).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from explainability_lab.narrative.candidate_context import build_candidate_context
from explainability_lab.narrative.grounded_narrative_generator import generate_narrative

FEATURE_STORE = "feature_lab/store/feature_store.jsonl"
CANDIDATES = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"


def normalise(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_shap_for_candidate(model, feature_store, candidate_id):
    """
    Compute SHAP top-5 for a single candidate.
    CRITICAL: use TRAINING_FEATURES (10 features), not FEATURE_NAMES_V2 (12).
    """
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

    feats = feature_store.get(candidate_id, {})
    feats_10 = TRAINING_FEATURES[:model.model.num_feature()]
    X = np.array([float(feats.get(f, 0) or 0) for f in feats_10])

    explainer = shap.TreeExplainer(model.model)
    shap_vals = explainer.shap_values(X.reshape(1, -1))[0]

    order = np.argsort(-np.abs(shap_vals))
    return [
        {"feature": feats_10[i], "shap_value": float(shap_vals[i])}
        for i in order[:5]
    ]


def run_prebatch():
    print("=== Pre-caching Narratives (Blend-Aware) ===")

    # 1. Load blend config
    alpha = 0.9
    blend_cfg_path = Path("ranking_lab/models/blend_config.json")
    if blend_cfg_path.exists():
        with open(blend_cfg_path) as f:
            cfg = json.load(f)
            alpha = cfg["alpha"]
    print(f"Using blend alpha: {alpha}")

    # 2. Load feature store
    store = {}
    with open(FEATURE_STORE) as f:
        for line in f:
            r = json.loads(line)
            store[r["candidate_id"]] = r
    print(f"Loaded {len(store)} candidates from feature store.")

    # 3. Load baseline model
    old_model = GBMLambdaRankModel()
    old_model.load("ranking_lab/models/gbm_lambdarank.txt")

    # 4. Predict baseline GBM scores
    cids = list(store.keys())
    feats_10 = [
        "skill_depth", "skill_breadth", "skill_recency", "skill_mastery_triangulation",
        "tenure_stability", "activity_quality_composite", "trust_score",
        "logistics_fit_score", "product_vs_services", "implied_skill_score"
    ]
    X_gbm = []
    for cid in cids:
        row = [float(store[cid].get(f, 0.0) or 0.0) for f in feats_10]
        X_gbm.append(row)
    X_gbm = np.array(X_gbm)
    
    gbm_scores = old_model.predict(X_gbm)
    gbm_scores_norm = normalise(gbm_scores)

    # 5. Extract and normalize new features
    jd_scores = np.array([float(store[c].get("jd_skill_score", 0) or 0) for c in cids])
    yoe_scores = np.array([float(store[c].get("yoe_band_fit", 0) or 0) for c in cids])
    jd_scores_norm = normalise(jd_scores)
    yoe_scores_norm = normalise(yoe_scores)
    new_scores = 0.6 * jd_scores_norm + 0.4 * yoe_scores_norm

    # 6. Compute final blend scores
    final_scores = alpha * gbm_scores_norm + (1.0 - alpha) * new_scores
    
    # Sort descending
    sorted_idx = np.argsort(-final_scores)
    top100_ids = [cids[i] for i in sorted_idx[:100]]
    top100_scores = [final_scores[i] for i in sorted_idx[:100]]

    # 7. Compute pool means for top-100
    pool_jd_mean = np.mean([float(store[cid].get("jd_skill_score", 0) or 0) for cid in top100_ids])
    pool_yoe_mean = np.mean([float(store[cid].get("yoe_band_fit", 0) or 0) for cid in top100_ids])
    print(f"Pool means (top-100) - jd_skill: {pool_jd_mean:.2f}, yoe_band: {pool_yoe_mean:.3f}")

    # 8. Load raw candidate profiles
    candidate_profiles = {}
    with open(CANDIDATES) as f:
        for line in f:
            c = json.loads(line)
            cid = c["candidate_id"]
            if cid in store:
                candidate_profiles[cid] = c

    # 9. Generate and cache narratives
    consolidated_cache = {}
    csv_rows = []
    print("Generating narratives for top-100 candidates...")
    
    # Keep cache dir to avoid re-generating from scratch
    cache_dir = Path("explainability_lab/narratives_cache")
            
    for rank_idx, cid in enumerate(tqdm(top100_ids)):
        rank = rank_idx + 1
        cand = candidate_profiles.get(cid)
        if not cand:
            continue
            
        features = store[cid]
        shap_contribs = compute_shap_for_candidate(old_model, store, cid)
            
        context = build_candidate_context(
            candidate=cand,
            features=features,
            shap_contributions=shap_contribs,
            rank=rank,
            pool_jd_skill_mean=pool_jd_mean,
            pool_yoe_mean=pool_yoe_mean
        )
        
        # generate_narrative writes candidate_id.json automatically inside explainability_lab/narratives_cache/
        narrative = generate_narrative(cid, context, mode="precompute")
        consolidated_cache[str(rank)] = {"narrative": narrative}
        
        csv_rows.append({
            "rank": rank,
            "candidate_id": cid,
            "score": top100_scores[rank_idx],
            "reasoning": narrative
        })

        import time
        time.sleep(1.0)

    # 10. Write consolidated cache for submission
    sub_dir = Path("submission")
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_cache_path = sub_dir / "narratives_cache.json"
    with open(sub_cache_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_cache, f, indent=2)
    print(f"Consolidated cache saved to {sub_cache_path}")

    # 11. Write submission.csv
    sub_csv_path = sub_dir / "submission.csv"
    pd.DataFrame(csv_rows).to_csv(sub_csv_path, index=False)
    print(f"Submission CSV saved to {sub_csv_path}")


if __name__ == "__main__":
    run_prebatch()
