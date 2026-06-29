# ranking_lab/experiments/exp_f_fix2.py
"""
Experiment F Fix-2: Replace activity_quality_composite with recruiter_response_rate.
Gate: NDCG@10 >= 0.7473
"""
import os, sys, json, datetime
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import lightgbm as lgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.models.feature_config_v2_fix2 import FEATURE_NAMES_FIX2, MONOTONIC_CONSTRAINTS_FIX2
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.evaluation.eval_metrics import compare_to_baseline

BASELINE = 0.7473
STORE_V2 = "feature_lab/store/feature_store_v2.jsonl"
STORE_FULL = "feature_lab/store/feature_store.jsonl"
LABELS   = "ranking_lab/labels/llm_labels.json"


def load_store():
    path_v2 = Path(STORE_V2)
    path_full = Path(STORE_FULL)
    filepath = path_full if (path_full.exists() and (not path_v2.exists() or path_full.stat().st_size > path_v2.stat().st_size)) else path_v2
    print(f"Loading feature store from: {filepath}")
    s = {}
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            s[r["candidate_id"]] = r
    return s


def load_labels():
    with open(LABELS) as f:
        raw = json.load(f)
    return {cid: int(v["label"] if isinstance(v, dict) else v) for cid, v in raw.items()}


def matrix(store, cids, feats):
    X, ids = [], []
    for cid in cids:
        if cid not in store: continue
        X.append([float(store[cid].get(f, 0.0) or 0.0) for f in feats])
        ids.append(cid)
    return np.array(X), ids


def train_eval(store, labels, gold, feats, constrs, seed):
    X_tr, tr_ids = matrix(store, list(labels.keys()), feats)
    y_tr = np.array([labels[c] for c in tr_ids])
    
    m = GBMLambdaRankModel(random_state=seed)
    m.features = feats
    m.constraints = constrs
    m.model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=100,
        learning_rate=0.05,
        monotone_constraints=constrs,
        random_state=seed,
        n_jobs=-1
    )
    
    m.fit(X_tr, y_tr)
    X_ev, ev_ids = matrix(store, list(gold.keys()), feats)
    preds = m.predict(X_ev)
    ranked = [ev_ids[i] for i in np.argsort(-preds)]
    return evaluate_ranking(ranked, gold)["ndcg@10"], m, ranked


def run():
    print("\n=== Experiment F Fix-2: Replace activity_quality_composite with recruiter_response_rate ===")
    print(f"Gate: NDCG@10 >= {BASELINE}\n")

    store  = load_store()
    labels = load_labels()
    gold   = load_gold_set()

    # 3-seed stability
    seed_results = {}
    for seed in [42, 7, 123]:
        ndcg, m, ranked = train_eval(
            store, labels, gold, FEATURE_NAMES_FIX2, MONOTONIC_CONSTRAINTS_FIX2, seed
        )
        seed_results[seed] = (ndcg, ranked)
        print(f"  seed={seed}: NDCG@10={ndcg:.4f}")

    # Spearman across seeds
    def rvec(ranked): return {c: i for i, c in enumerate(ranked)}
    common = list(set(seed_results[42][1]) & set(seed_results[7][1]) & set(seed_results[123][1]))
    r42  = [rvec(seed_results[42][1])[c]  for c in common]
    r7   = [rvec(seed_results[7][1])[c]   for c in common]
    r123 = [rvec(seed_results[123][1])[c] for c in common]
    sp_42_7,   _ = scipy_stats.spearmanr(r42, r7)
    sp_42_123, _ = scipy_stats.spearmanr(r42, r123)
    print(f"  Spearman r (42 vs 7):   {sp_42_7:.4f}")
    print(f"  Spearman r (42 vs 123): {sp_42_123:.4f}")

    if sp_42_7 == 1.0 and sp_42_123 == 1.0:
        print("  [Warning] Spearman still 1.0 - model still collapsed. Proceed to Fix-3.")

    # Primary model
    primary_ndcg, primary_m, primary_ranked = train_eval(
        store, labels, gold, FEATURE_NAMES_FIX2, MONOTONIC_CONSTRAINTS_FIX2, 42
    )
    imps = primary_m.model.feature_importances_
    total_imp = imps.sum()
    normalized_imps = imps / total_imp if total_imp > 0 else imps
    
    print("\n  Feature Importances:")
    for feat, imp in sorted(zip(FEATURE_NAMES_FIX2, normalized_imps), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    # Gate
    comp = compare_to_baseline({"ndcg@10": primary_ndcg}, BASELINE)
    print(f"\n  Gate: {comp['verdict']} (NDCG={primary_ndcg:.4f}, delta={comp['delta']:+.4f})")

    # Netflix spot-check
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
            
    print("\n  Netflix/Aganitha spot-check:")
    for i, cid in enumerate(primary_ranked[:120]):
        co = co_map.get(cid,"?")
        if any(x in co.lower() for x in ["netflix","aganitha"]):
            print(f"    Rank {i+1}: {cid} @ {co}")

    if comp["verdict"] != "REGRESSION":
        primary_m.save("ranking_lab/models/gbm_lambdarank_v2.txt")
        print("\n  [OK] Gate passed. Model saved as gbm_lambdarank_v2.txt")
        print("  -> Proceed to PROMPT_03A (ablation)")
    else:
        print("\n  [FAIL] Fix-2 failed. Proceed to Fix-3.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report = {
        "experiment": "exp_f_fix2",
        "features": FEATURE_NAMES_FIX2,
        "ndcg@10": primary_ndcg,
        "comparison": comp,
        "spearman_42_7": sp_42_7,
        "spearman_42_123": sp_42_123,
        "feature_importances": dict(zip(FEATURE_NAMES_FIX2, imps.tolist())),
    }
    out = Path(f"reports_archive/exp_f_fix2_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {out}")
    return report


if __name__ == "__main__":
    run()
