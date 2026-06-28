"""
Experiment F: Retrain LambdaRank v2 (12 features).
GATE: NDCG@10 >= 0.7473
"""
import os, sys, json, datetime
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.models.feature_config_v2 import FEATURE_NAMES_V2
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.evaluation.eval_metrics import compare_to_baseline

BASELINE_NDCG10 = 0.7473
FEATURE_STORE = "feature_lab/store/feature_store.jsonl"
LABELS_PATH   = "ranking_lab/labels/llm_labels.json"
MODEL_OUT     = "ranking_lab/models/gbm_lambdarank_v2.txt"


def load_store():
    path_v2 = Path("feature_lab/store/feature_store_v2.jsonl")
    path_full = Path("feature_lab/store/feature_store.jsonl")
    
    # We want to use the one with the features, which is feature_store.jsonl
    filepath = path_full if path_full.stat().st_size > path_v2.stat().st_size else path_v2
    print(f"Loading feature store from: {filepath}")
    
    store = {}
    with open(filepath) as f:
        for line in f:
            row = json.loads(line)
            store[row["candidate_id"]] = row
    return store


def load_labels():
    with open(LABELS_PATH) as f:
        raw = json.load(f)
    return {cid: int(v["label"] if isinstance(v, dict) else v) for cid, v in raw.items()}


def matrix(store, cids, feats):
    X, ids = [], []
    for cid in cids:
        if cid not in store:
            continue
        X.append([float(store[cid].get(f, 0.0) or 0.0) for f in feats])
        ids.append(cid)
    return np.array(X), ids


def train_eval(store, labels, gold, seed):
    X_tr, tr_ids = matrix(store, list(labels.keys()), FEATURE_NAMES_V2)
    y_tr = np.array([labels[c] for c in tr_ids])
    
    m = GBMLambdaRankModel(random_state=seed)
    m.fit(X_tr, y_tr)
    X_ev, ev_ids = matrix(store, list(gold.keys()), FEATURE_NAMES_V2)
    preds = m.predict(X_ev)
    ranked = [ev_ids[i] for i in np.argsort(-preds)]
    ndcg = evaluate_ranking(ranked, gold)["ndcg@10"]
    return ndcg, m, ranked, preds, ev_ids


def run():
    print("\n=== Experiment F: LambdaRank v2 Retrain ===")
    store  = load_store()
    labels = load_labels()
    gold   = load_gold_set()
    print(f"Training on {len(labels)} labeled | Evaluating on {len(gold)} gold")

    # 3-seed stability
    print("\n--- Stability (3 seeds) ---")
    seed_ranked = {}
    for seed in [42, 7, 123]:
        ndcg, m, ranked, preds, ev_ids = train_eval(store, labels, gold, seed)
        seed_ranked[seed] = ranked
        print(f"  seed={seed}: NDCG@10={ndcg:.4f}")

    # Spearman across seeds
    def rank_vec(ranked):
        return {cid: i for i, cid in enumerate(ranked)}
    common = list(set(seed_ranked[42]) & set(seed_ranked[7]) & set(seed_ranked[123]))
    r42  = [rank_vec(seed_ranked[42])[c]  for c in common]
    r7   = [rank_vec(seed_ranked[7])[c]   for c in common]
    r123 = [rank_vec(seed_ranked[123])[c] for c in common]
    sp_42_7,   _ = scipy_stats.spearmanr(r42, r7)
    sp_42_123, _ = scipy_stats.spearmanr(r42, r123)
    print(f"  Spearman r (42 vs 7):   {sp_42_7:.4f}")
    print(f"  Spearman r (42 vs 123): {sp_42_123:.4f}")

    # Primary model (seed=42)
    primary_ndcg, primary_model, primary_ranked, primary_preds, primary_ev_ids = \
        train_eval(store, labels, gold, 42)

    # Feature importances
    importances = primary_model.model.feature_importances_
    total_imp = importances.sum()
    normalized_importances = importances / total_imp if total_imp > 0 else importances
    
    print("\n--- Feature Importances ---")
    pairs = sorted(zip(FEATURE_NAMES_V2, normalized_importances), key=lambda x: -x[1])
    for feat, imp in pairs:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    # Comparison
    comp = compare_to_baseline({"ndcg@10": primary_ndcg}, BASELINE_NDCG10)
    print(f"\n--- Gate Check ---")
    print(f"  Baseline: {BASELINE_NDCG10}  |  New: {primary_ndcg:.4f}  "
          f"|  Delta: {comp['delta']:+.4f}  |  Verdict: {comp['verdict']}")

    # Spot-check Netflix vs Aganitha concern
    print("\n--- Spot-Check: Netflix (deep-IR) vs Aganitha (buzzword) ---")
    cand_path = next(Path("dataset").rglob("sample_candidates.json"), None)
    if cand_path:
        with open(cand_path) as f:
            raw = json.load(f)
        company_map = {}
        for c in raw:
            cur = next((r for r in c.get("career_history", [])
                        if r.get("is_current")), {})
            company_map[c["candidate_id"]] = cur.get("company", "?")
        for i, cid in enumerate(primary_ranked[:60]):
            co = company_map.get(cid, "?")
            if any(x in co.lower() for x in ["netflix", "aganitha"]):
                print(f"  Rank {i+1}: {cid} @ {co}")

    # Save
    if comp["verdict"] != "REGRESSION":
        primary_model.save(MODEL_OUT)
        print(f"\n[OK] Model saved: {MODEL_OUT}")
    else:
        print(f"\n[GATE FAILED] model NOT saved. Stop and report before proceeding.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report = {
        "experiment": "exp_f_retrain_v2",
        "ndcg@10": primary_ndcg,
        "comparison": comp,
        "stability": {
            "seeds": {s: evaluate_ranking(r, gold)["ndcg@10"]
                      for s, r in seed_ranked.items()},
            "spearman_42_7": sp_42_7,
            "spearman_42_123": sp_42_123,
        },
        "feature_importances": dict(zip(FEATURE_NAMES_V2, importances.tolist())),
    }
    out = Path(f"reports_archive/exp_f_retrain_v2_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")
    return report


if __name__ == "__main__":
    run()
