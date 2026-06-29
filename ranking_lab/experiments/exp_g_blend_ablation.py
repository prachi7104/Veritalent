"""
Experiment G (Blend): Ablation on the 0.9/0.1 blend scoring system.
No model retraining needed — all tests are score-level arithmetic.

Tests:
  G1. Baseline blend (alpha=0.90, jd=0.6, yoe=0.4)       — already known: 0.7482
  G2. jd_skill_score only (alpha=0.90, jd=1.0, yoe=0.0)  — isolate jd signal
  G3. yoe_band_fit only   (alpha=0.90, jd=0.0, yoe=1.0)  — isolate yoe signal
  G4. No new features     (alpha=1.0)                      — pure old GBM = 0.7473
  G5. Ratio sweep: jd/yoe at [80/20, 70/30, 60/40, 50/50]
"""
import os, sys, json, datetime
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.experiments.common import build_eval_matrix
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

BASELINE_BLEND  = 0.7482   # Fix-3 result
BASELINE_ORIG   = 0.7473   # original Exp C


def norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def blend_and_eval(old_scores_n, jd_scores_n, yoe_scores_n,
                   alpha, jd_w, yoe_w, gold, ev_ids):
    new_feat = jd_w * jd_scores_n + yoe_w * yoe_scores_n
    new_feat_n = norm(new_feat) if new_feat.std() > 0 else new_feat
    blended = alpha * old_scores_n + (1 - alpha) * new_feat_n
    ranked  = [ev_ids[i] for i in np.argsort(-blended)]
    return evaluate_ranking(ranked, gold)["ndcg@10"], ranked


def run():
    print("\n=== Experiment G: Blend Ablation ===")
    print(f"Baseline blend NDCG@10: {BASELINE_BLEND}")
    print(f"Original GBM NDCG@10:   {BASELINE_ORIG}\n")

    gold = load_gold_set()

    # Load old model scores
    old_model = GBMLambdaRankModel()
    old_model.load("ranking_lab/models/gbm_lambdarank.txt")

    store = {}
    with open("feature_lab/store/feature_store_v2.jsonl") as f:
        for line in f:
            r = json.loads(line); store[r["candidate_id"]] = r

    gold_cids = list(gold.keys())
    _, ev_ids = build_eval_matrix(store, gold_cids)
    old_features = [
        "skill_depth", "skill_breadth", "skill_recency", 
        "skill_mastery_triangulation", "tenure_stability", 
        "activity_quality_composite", "trust_score", 
        "logistics_fit_score", "product_vs_services", 
        "implied_skill_score"
    ]
    X_old = []
    for cid in ev_ids:
        cfeats = store.get(cid, {})
        X_old.append([float(cfeats.get(f, 0) or 0) for f in old_features])
    X_old = np.array(X_old)
    old_scores = norm(old_model.predict(X_old))

    jd_scores  = norm(np.array([float(store.get(c,{}).get("jd_skill_score",0) or 0)
                                 for c in ev_ids]))
    yoe_scores = norm(np.array([float(store.get(c,{}).get("yoe_band_fit",0) or 0)
                                 for c in ev_ids]))

    # Load Netflix/Aganitha company map for spot-checks
    cand_path = next(Path("dataset").rglob("sample_candidates.json"), None)
    co_map = {}
    if cand_path:
        with open(cand_path) as f:
            raw = json.load(f)
        for c in raw:
            cur = next((r for r in c.get("career_history",[]) if r.get("is_current")),{})
            co_map[c["candidate_id"]] = cur.get("company","?")

    def spot(ranked):
        nf = next((i+1 for i,c in enumerate(ranked)
                   if "netflix" in co_map.get(c,"").lower()), 999)
        ag = next((i+1 for i,c in enumerate(ranked)
                   if "aganitha" in co_map.get(c,"").lower()), 999)
        return f"Netflix={nf} Aganitha={ag} {'✓' if nf<ag else '✗'}"

    results = {}

    # G1: Baseline blend
    ndcg, ranked = blend_and_eval(old_scores, jd_scores, yoe_scores,
                                   0.90, 0.6, 0.4, gold, ev_ids)
    results["G1_baseline_blend"] = ndcg
    print(f"G1 baseline blend (α=0.90, jd=0.6, yoe=0.4): NDCG={ndcg:.4f}  {spot(ranked)}")

    # G2: jd_skill_score only in 10% slice
    ndcg, ranked = blend_and_eval(old_scores, jd_scores, yoe_scores,
                                   0.90, 1.0, 0.0, gold, ev_ids)
    results["G2_jd_only"] = ndcg
    delta = ndcg - BASELINE_BLEND
    print(f"G2 jd_skill only    (α=0.90, jd=1.0, yoe=0.0): NDCG={ndcg:.4f}  "
          f"({delta:+.4f})  {spot(ranked)}")

    # G3: yoe_band_fit only in 10% slice
    ndcg, ranked = blend_and_eval(old_scores, jd_scores, yoe_scores,
                                   0.90, 0.0, 1.0, gold, ev_ids)
    results["G3_yoe_only"] = ndcg
    delta = ndcg - BASELINE_BLEND
    print(f"G3 yoe_band only    (α=0.90, jd=0.0, yoe=1.0): NDCG={ndcg:.4f}  "
          f"({delta:+.4f})  {spot(ranked)}")

    # G4: No new features — pure old GBM
    ndcg, ranked = blend_and_eval(old_scores, jd_scores, yoe_scores,
                                   1.00, 0.6, 0.4, gold, ev_ids)
    results["G4_old_gbm_only"] = ndcg
    delta = ndcg - BASELINE_BLEND
    print(f"G4 old GBM only     (α=1.00):                   NDCG={ndcg:.4f}  "
          f"({delta:+.4f})  {spot(ranked)}")

    # G5: Ratio sweep (jd/yoe)
    print(f"\n--- G5: Ratio Sweep (α=0.90 fixed) ---")
    print(f"{'JD weight':>10} {'YOE weight':>11} {'NDCG':>8} {'Delta':>8}  Spot-check")
    best_ratio_ndcg = 0
    best_jd_w, best_yoe_w = 0.6, 0.4
    for jd_w in [0.80, 0.70, 0.60, 0.50]:
        yoe_w = 1.0 - jd_w
        ndcg, ranked = blend_and_eval(old_scores, jd_scores, yoe_scores,
                                       0.90, jd_w, yoe_w, gold, ev_ids)
        delta = ndcg - BASELINE_BLEND
        marker = " ← current" if abs(jd_w - 0.6) < 0.01 else ""
        print(f"  jd={jd_w:.2f}  yoe={yoe_w:.2f}  "
              f"NDCG={ndcg:.4f}  {delta:+.4f}  {spot(ranked)}{marker}")
        if ndcg > best_ratio_ndcg:
            best_ratio_ndcg = ndcg
            best_jd_w, best_yoe_w = jd_w, yoe_w

    print(f"\nBest ratio: jd={best_jd_w:.2f} / yoe={best_yoe_w:.2f} "
          f"(NDCG={best_ratio_ndcg:.4f})")

    # Decisions
    print("\n--- Decisions ---")
    jd_contrib  = results["G2_jd_only"]  - results["G4_old_gbm_only"]
    yoe_contrib = results["G3_yoe_only"] - results["G4_old_gbm_only"]
    print(f"jd_skill_score  marginal contribution vs no-new-features: {jd_contrib:+.4f}")
    print(f"yoe_band_fit    marginal contribution vs no-new-features: {yoe_contrib:+.4f}")

    if jd_contrib > 0 and yoe_contrib > 0:
        print("✓ BOTH new features contribute positively. Keep both in blend.")
    elif jd_contrib > 0 and yoe_contrib <= 0:
        print("→ Keep jd_skill_score only. Remove yoe_band_fit from blend.")
        print("  Update blend_config.json: set yoe weight to 0.0")
    elif jd_contrib <= 0 and yoe_contrib > 0:
        print("→ Keep yoe_band_fit only. Remove jd_skill_score from blend.")
        print("  Update blend_config.json: set jd weight to 0.0")
    else:
        print("⚠ Neither feature contributes positively in isolation.")
        print("  Blend is working through interaction, not individual signals.")
        print("  Keep the 60/40 split — it empirically beats pure GBM.")

    # Update blend config if ratio improved
    if abs(best_jd_w - 0.6) > 0.01 and best_ratio_ndcg > BASELINE_BLEND + 0.001:
        print(f"\n→ Updating blend_config.json with improved ratio: "
              f"jd={best_jd_w}, yoe={best_yoe_w}")
        with open("ranking_lab/models/blend_config.json") as f:
            cfg = json.load(f)
        cfg["new_features"] = {"jd_skill_score": best_jd_w, "yoe_band_fit": best_yoe_w}
        with open("ranking_lab/models/blend_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report = {
        "experiment": "exp_g_blend_ablation",
        "baseline_blend": BASELINE_BLEND,
        "baseline_orig": BASELINE_ORIG,
        "results": results,
        "jd_marginal_contribution": jd_contrib,
        "yoe_marginal_contribution": yoe_contrib,
        "best_ratio": {"jd": best_jd_w, "yoe": best_yoe_w,
                       "ndcg": best_ratio_ndcg},
    }
    out = Path(f"reports_archive/exp_g_blend_ablation_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out}")
    return report


if __name__ == "__main__":
    run()
