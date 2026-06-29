"""
Experiment D2: Cross-Encoder Reranker — three configurations.

D1: GBM LambdaRank only (baseline, NDCG@10=0.7473)
D2: GBM → CE reranks top-50
D3: GBM → CE reranks top-20 only (faster, less disruption)

Gate: D2 or D3 must beat 0.7473 AND latency < 2000ms.

IMPORTANT: This experiment uses the OLD gbm_lambdarank.txt for comparison
purity. After PROMPT_02A we can optionally re-run with gbm_lambdarank_v2.txt
but that's a bonus — not required for the gate decision.
"""
import os, sys, json, datetime
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from retrieval_lab.rerankers.cross_encoder_reranker import CrossEncoderReranker
from ranking_lab.evaluation.ndcg_eval import load_gold_set, evaluate_ranking
from ranking_lab.evaluation.eval_metrics import compare_to_baseline

BASELINE_NDCG10 = 0.7473
LATENCY_GATE_MS = 2000

JD_TEXT = """
Senior AI/ML Engineer — Search and Retrieval

We are building next-generation search infrastructure. You will own our
embedding pipelines, ranking models, and large-scale retrieval systems.

Requirements:
- 5-9 years of experience in AI/ML
- Deep expertise in information retrieval: BM25, dense retrieval, vector search
- Experience with learning-to-rank (LambdaMART, LTR frameworks)
- Proficiency with embedding models (BERT, Sentence Transformers, E5, BGE)
- Hands-on with search backends: Elasticsearch, Qdrant, Pinecone, Weaviate
- Strong Python, PyTorch preferred
- Product company experience preferred
- Location: Pune or Noida (hybrid)
- Notice period: 30 days or less preferred
"""


def load_all_candidates():
    cand_path = next(Path("dataset").rglob("sample_candidates.json"), None)
    if not cand_path:
        raise FileNotFoundError("Dataset not found")
    with open(cand_path) as f:
        raw = json.load(f)
    return {c["candidate_id"]: c for c in raw}


def get_gbm_ranking(store, gold, model_path):
    """Get ranked list from the GBM model."""
    from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
    from ranking_lab.experiments.common import build_eval_matrix
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

    model = GBMLambdaRankModel()
    model.load(model_path)
    X_ev, ev_ids = build_eval_matrix(store, list(gold.keys()))
    preds = model.predict(X_ev)
    return [ev_ids[i] for i in np.argsort(-preds)]


def run():
    print("\n=== Experiment D2: Cross-Encoder Reranker ===")
    gold = load_gold_set()
    cand_map = load_all_candidates()

    # Feature store for GBM
    feat_store = {}
    store_path = Path("feature_lab/store/feature_store_v2.jsonl")
    if not store_path.exists():
        store_path = Path("feature_lab/store/feature_store.jsonl")
    with open(store_path) as f:
        for line in f:
            row = json.loads(line)
            feat_store[row["candidate_id"]] = row

    # --- D1: GBM baseline ---
    model_path = "ranking_lab/models/gbm_lambdarank_v2.txt"
    if not Path(model_path).exists():
        model_path = "ranking_lab/models/gbm_lambdarank.txt"
    print(f"\nUsing GBM model: {model_path}")

    gbm_ranked = get_gbm_ranking(feat_store, gold, model_path)
    d1_metrics = evaluate_ranking(gbm_ranked, gold)
    print(f"D1 (GBM baseline): NDCG@10={d1_metrics['ndcg@10']:.4f}")

    reranker = CrossEncoderReranker()
    candidates_ordered = [cand_map.get(cid, {}) for cid in gbm_ranked]

    results = {"D1_gbm_baseline": {"ndcg@10": d1_metrics["ndcg@10"], "latency_ms": 0}}

    # --- D2: CE reranks top-50 ---
    print("\nD2: CE reranks top-50...")
    ce50_ids, ce50_scores, lat50 = reranker.rerank(
        JD_TEXT, candidates_ordered[:50], gbm_ranked[:50], top_n=50
    )
    d2_full = ce50_ids + gbm_ranked[50:]
    d2_metrics = evaluate_ranking(d2_full, gold)
    print(f"D2 (CE top-50): NDCG@10={d2_metrics['ndcg@10']:.4f}, latency={lat50:.0f}ms")
    results["D2_ce_top50"] = {"ndcg@10": d2_metrics["ndcg@10"], "latency_ms": lat50}

    # --- D3: CE reranks top-20 ---
    print("D3: CE reranks top-20...")
    ce20_ids, ce20_scores, lat20 = reranker.rerank(
        JD_TEXT, candidates_ordered[:20], gbm_ranked[:20], top_n=20
    )
    d3_full = ce20_ids + gbm_ranked[20:]
    d3_metrics = evaluate_ranking(d3_full, gold)
    print(f"D3 (CE top-20): NDCG@10={d3_metrics['ndcg@10']:.4f}, latency={lat20:.0f}ms")
    results["D3_ce_top20"] = {"ndcg@10": d3_metrics["ndcg@10"], "latency_ms": lat20}

    # --- Summary and gate ---
    print("\n=== SUMMARY ===")
    best_config = max(
        [k for k in results if k != "D1_gbm_baseline"],
        key=lambda k: results[k]["ndcg@10"]
    )
    best = results[best_config]
    delta = best["ndcg@10"] - BASELINE_NDCG10

    for name, r in results.items():
        d = r["ndcg@10"] - BASELINE_NDCG10
        v = "IMPROVEMENT" if d > 0.005 else "REGRESSION" if d < -0.005 else "NEUTRAL"
        lat = f"{r['latency_ms']:.0f}ms" if r['latency_ms'] else "N/A"
        print(f"  {name:<25} NDCG@10={r['ndcg@10']:.4f}  ({v:>11}, {d:+.4f})  lat={lat}")

    gate_ndcg = best["ndcg@10"] > BASELINE_NDCG10
    gate_lat  = best["latency_ms"] < LATENCY_GATE_MS
    print(f"\nGate: NDCG {'PASS' if gate_ndcg else 'FAIL'} | "
          f"Latency {'PASS' if gate_lat else 'FAIL'}")

    if gate_ndcg and gate_lat:
        print(f"✓ INTEGRATE: Use {best_config} in final pipeline.")
        recommendation = f"integrate_{best_config}"
    else:
        print("✗ DO NOT INTEGRATE: CE adds latency without NDCG gain.")
        print("  Final pipeline: GBM LambdaRank only.")
        recommendation = "exclude_cross_encoder"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    report = {"results": results, "recommendation": recommendation,
              "gate_ndcg_pass": gate_ndcg, "gate_latency_pass": gate_lat}
    out = Path(f"reports_archive/exp_d2_cross_encoder_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")
    return report


if __name__ == "__main__":
    run()
