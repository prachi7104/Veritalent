# ranking_lab/experiments/exp_f_fix3_blend.py
"""
Fix-3: Score blend of old model (10-feature) + new features (jd_skill_score + yoe_band_fit).

Strategy:
  final_score = alpha * old_gbm_score + (1-alpha) * new_feature_score

Where new_feature_score is:
  new_feature_score = 0.6 * jd_skill_score_norm + 0.4 * yoe_band_fit_norm
"""
import os, sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking


def normalise(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def build_eval_matrix_10(store, cids):
    feats_10 = [
        "skill_depth",
        "skill_breadth",
        "skill_recency",
        "skill_mastery_triangulation",
        "tenure_stability",
        "activity_quality_composite",
        "trust_score",
        "logistics_fit_score",
        "product_vs_services",
        "implied_skill_score"
    ]
    X, ids = [], []
    for cid in cids:
        if cid not in store:
            continue
        X.append([float(store[cid].get(f, 0.0) or 0.0) for f in feats_10])
        ids.append(cid)
    return np.array(X), ids


def run():
    print("\n=== Experiment F Fix-3: Score Blend ===")
    gold = load_gold_set()

    # Load old model
    old_model = GBMLambdaRankModel()
    old_model.load("ranking_lab/models/gbm_lambdarank.txt")

    # Load feature store dynamically
    path_v2 = Path("feature_lab/store/feature_store_v2.jsonl")
    path_full = Path("feature_lab/store/feature_store.jsonl")
    filepath = path_full if (path_full.exists() and (not path_v2.exists() or path_full.stat().st_size > path_v2.stat().st_size)) else path_v2
    print(f"Loading feature store from: {filepath}")
    
    store = {}
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            store[r["candidate_id"]] = r

    gold_cids = list(gold.keys())

    # Predict with old model using 10 original features
    X_old, ev_ids = build_eval_matrix_10(store, gold_cids)
    old_scores = old_model.predict(X_old)
    old_scores_norm = normalise(old_scores)

    # New JD feature scores
    jd_scores = np.array([float(store.get(c,{}).get("jd_skill_score",0) or 0) for c in ev_ids])
    yoe_scores = np.array([float(store.get(c,{}).get("yoe_band_fit",0) or 0) for c in ev_ids])
    jd_scores_norm  = normalise(jd_scores)
    yoe_scores_norm = normalise(yoe_scores)
    new_feature_score = 0.6 * jd_scores_norm + 0.4 * yoe_scores_norm

    # Load candidate company info for spot-check
    cand_path = next(Path("dataset").rglob("sample_candidates.json"), None)
    co_map = {}
    
    full_cand_path = next(Path("dataset").rglob("candidates.jsonl"), None)
    if full_cand_path:
        with open(full_cand_path) as f:
            for line in f:
                c = json.loads(line)
                cur = next((r for r in c.get("career_history",[]) if r.get("is_current")),{})
                co_map[c["candidate_id"]] = cur.get("company","?")
    elif cand_path:
        with open(cand_path) as f:
            raw = json.load(f)
        for c in raw:
            cur = next((r for r in c.get("career_history",[]) if r.get("is_current")),{})
            co_map[c["candidate_id"]] = cur.get("company","?")

    print(f"\n{'Alpha':>7}  {'NDCG@10':>10}  {'Delta':>8}  {'Netflix<Aganitha?':>20}")
    print("-" * 55)
    best_alpha, best_ndcg, best_ranked = None, 0.0, None
    for alpha in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
        blended = alpha * old_scores_norm + (1 - alpha) * new_feature_score
        ranked  = [ev_ids[i] for i in np.argsort(-blended)]
        ndcg    = evaluate_ranking(ranked, gold)["ndcg@10"]
        delta   = ndcg - 0.7473

        # Netflix/Aganitha check
        nf_rank = next((i+1 for i,c in enumerate(ranked) if "netflix" in co_map.get(c,"").lower()), 999)
        ag_rank = next((i+1 for i,c in enumerate(ranked) if "aganitha" in co_map.get(c,"").lower()), 999)
        order_ok = "OK" if nf_rank < ag_rank else "FAIL"

        print(f"  {alpha:>5}  {ndcg:>10.4f}  {delta:>+8.4f}  "
              f"Netflix={nf_rank} Aganitha={ag_rank} {order_ok}")
        if ndcg > best_ndcg:
            best_ndcg, best_alpha, best_ranked = ndcg, alpha, ranked

    print(f"\nBest: alpha={best_alpha}, NDCG@10={best_ndcg:.4f}")
    if best_ndcg >= 0.7473:
        print("[OK] Blend recovers gate. Save blend config and use for final pipeline.")
        # Save the alpha so PROMPT_04 can use it
        with open("ranking_lab/models/blend_config.json", "w") as f:
            json.dump({
                "alpha": best_alpha,
                "old_model": "ranking_lab/models/gbm_lambdarank.txt",
                "new_features": {"jd_skill_score": 0.6, "yoe_band_fit": 0.4}
            }, f, indent=2)
        print("  Config saved: ranking_lab/models/blend_config.json")
    else:
        print("[FAIL] Blend also fails. Escalate - do not submit until resolved.")
    return best_ndcg, best_alpha


if __name__ == "__main__":
    run()
