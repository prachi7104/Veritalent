"""
marginal_value_test.py — NDCG lift from the fingerprint feature.

Measures NDCG@50 delta: (linear scorer + fingerprint bonus) vs. (linear scorer alone),
holding all other features constant. Uses the feature_lab feature store and gold_set_pooled.json.

Given n=8 fingerprint holders out of 100k candidates, the aggregate NDCG delta
is expected to be small. This is reported honestly.

Usage:
    python -m fingerprint_lab.analysis.marginal_value_test \\
        --store  <path/to/feature_store.jsonl> \\
        --audit  <path/to/frequency_audit_results.json>
"""
import json
import math
import argparse
import os
from collections import defaultdict
from pathlib import Path

# Gold set config — mirrors ablation_runner.py
GOLD_SET_POOLED = r"C:\projects\Veritalent\retrieval_lab\evaluation\gold_set_pooled.json"
GOLD_SET_FALLBACK = r"C:\projects\Veritalent\retrieval_lab\evaluation\gold_set.json"

FEATURE_GROUPS = {
    "skill":     ["skill_depth", "skill_breadth", "skill_recency", "skill_mastery_triangulation"],
    "career":    ["career_velocity", "promotion_velocity", "tenure_stability", "inflection_point_strength"],
    "trust":     ["trust_score"],
    "activity":  ["activity_quality_composite"],
    "industry":  ["industry_relevance"],
    "logistics": ["logistics_fit_score"],
    "company":   ["product_vs_services"],
}
GROUP_WEIGHTS = {
    "skill": 0.30, "career": 0.15, "activity": 0.20,
    "industry": 0.10, "company": 0.10, "trust": 0.10, "logistics": 0.05,
}


def get_gold_path() -> str:
    return GOLD_SET_POOLED if os.path.exists(GOLD_SET_POOLED) else GOLD_SET_FALLBACK


def compute_ndcg_at_k(ranked_ids: list, judgments: dict, k: int = 50) -> float:
    dcg = sum(
        (2 ** judgments.get(cid, 0) - 1) / math.log2(i + 2)
        for i, cid in enumerate(ranked_ids[:k])
    )
    ideal_rels = sorted(judgments.values(), reverse=True)
    idcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(ideal_rels[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0


def load_feature_store(path: str) -> dict:
    candidates = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                candidates[r["candidate_id"]] = r
    return candidates


def normalize_features(candidates: dict) -> None:
    min_v: dict = defaultdict(lambda: float("inf"))
    max_v: dict = defaultdict(lambda: float("-inf"))
    for feats in candidates.values():
        for fs in FEATURE_GROUPS.values():
            for f in fs:
                v = float(feats.get(f, 0.0) or 0.0)
                if v < min_v[f]: min_v[f] = v
                if v > max_v[f]: max_v[f] = v
    for feats in candidates.values():
        for fs in FEATURE_GROUPS.values():
            for f in fs:
                v = float(feats.get(f, 0.0) or 0.0)
                d = max_v[f] - min_v[f]
                feats[f"{f}_norm"] = (v - min_v[f]) / d if d > 0 else 0.0


def base_score(feats: dict) -> float:
    s = 0.0
    for group, fs in FEATURE_GROUPS.items():
        s += sum(feats.get(f"{f}_norm", 0.0) for f in fs) * GROUP_WEIGHTS[group]
    return s


def run_marginal_value_test(
    store_path: str,
    audit_path: str | None = None,
    fp_bonus: float = 0.1,
) -> dict:
    print("Loading feature store...")
    candidates = load_feature_store(store_path)
    normalize_features(candidates)

    # Get fp holder IDs
    fp_holder_ids: set[str] = set()
    if audit_path and Path(audit_path).exists():
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        fp_holder_ids = {d["candidate_id"] for d in audit.get("fp_holder_details", [])}
        print(f"Fingerprint holders loaded from audit: {len(fp_holder_ids)}")
    else:
        print("Warning: no audit file — fingerprint holders unknown, delta will be 0")

    gold_path = get_gold_path()
    print(f"Loading gold set from {gold_path}...")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)
    judgments = gold["queries"][0]["judgments"]

    # Score WITHOUT fingerprint bonus
    scores_no_fp = [(cid, base_score(feats)) for cid, feats in candidates.items()]
    scores_no_fp.sort(key=lambda x: x[1], reverse=True)
    ndcg_no_fp = compute_ndcg_at_k([x[0] for x in scores_no_fp], judgments)

    # Score WITH fingerprint bonus
    scores_fp = [
        (cid, base_score(feats) + (fp_bonus if cid in fp_holder_ids else 0.0))
        for cid, feats in candidates.items()
    ]
    scores_fp.sort(key=lambda x: x[1], reverse=True)
    ndcg_fp = compute_ndcg_at_k([x[0] for x in scores_fp], judgments)

    delta = ndcg_fp - ndcg_no_fp

    # Which fp holders are in gold set and at what relevance?
    fp_in_gold = {cid: judgments[cid] for cid in fp_holder_ids if cid in judgments}

    # Rank of fp holders in both rankers
    rank_no_fp = {x[0]: i + 1 for i, x in enumerate(scores_no_fp)}
    rank_fp    = {x[0]: i + 1 for i, x in enumerate(scores_fp)}
    fp_rank_detail = [
        {
            "candidate_id": cid,
            "relevance_in_gold": fp_in_gold.get(cid, "NOT IN GOLD"),
            "rank_without_fp":   rank_no_fp[cid],
            "rank_with_fp":      rank_fp[cid],
        }
        for cid in fp_holder_ids
    ]
    fp_rank_detail.sort(key=lambda x: x["rank_without_fp"])

    results = {
        "ndcg_at_50_without_fingerprint": round(ndcg_no_fp, 4),
        "ndcg_at_50_with_fingerprint":    round(ndcg_fp, 4),
        "ndcg_delta":                     round(delta, 4),
        "fp_bonus_applied":               fp_bonus,
        "fp_holder_count":                len(fp_holder_ids),
        "fp_holders_in_gold_set":         fp_in_gold,
        "fp_rank_detail":                 fp_rank_detail,
        "gold_set_path_used":             gold_path,
    }

    print(f"\n=== MARGINAL VALUE TEST RESULTS ===")
    print(f"NDCG@50 without fingerprint: {ndcg_no_fp:.4f}")
    print(f"NDCG@50 with fingerprint:    {ndcg_fp:.4f}")
    print(f"Delta:                       {delta:+.4f}")
    print(f"\nFingerprint holders in gold set: {len(fp_in_gold)}/{len(fp_holder_ids)}")
    print(f"Their relevance scores: {fp_in_gold}")
    print(f"\nRank detail:")
    for r in fp_rank_detail:
        print(f"  {r['candidate_id']}: rank {r['rank_without_fp']:,} -> {r['rank_with_fp']:,} "
              f"(gold rel: {r['relevance_in_gold']})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store",  required=True, help="Path to feature_store.jsonl")
    parser.add_argument("--audit",  default=None,  help="Path to frequency_audit_results.json")
    parser.add_argument("--bonus",  type=float, default=0.1, help="Fingerprint score bonus")
    args = parser.parse_args()
    run_marginal_value_test(args.store, args.audit, args.bonus)
