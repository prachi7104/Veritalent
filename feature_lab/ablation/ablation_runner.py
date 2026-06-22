"""
ablation_runner.py — Feature group ablation testing.

Methodology (approved, per design decision 2026-06-22):
  Uses a linear weighted scoring formula (NOT LightGBM/cross-validation).
  Reasons:
    - Gold set has ~200 candidates. Training a GBM on 80 rows per fold
      produces noise, not signal.
    - Linear scoring is deterministic, interpretable, and directly isolates
      feature contribution without training randomness.
    - LightGBM training with lambdarank belongs in Lab 06, not here.

Scoring formula:
  1. Min-max normalize each feature across the full 100k pool.
  2. Sum normalized features within each group.
  3. Weight each group by GROUP_WEIGHTS (normalized to sum 1.0).
  4. Rank all 100k candidates by final score.
  5. Measure NDCG@50 against gold-set judgments.

Ablation:
  Remove one feature group at a time, re-score, re-rank, re-measure NDCG@50.
  Delta = NDCG_without - NDCG_baseline.

Gold set path:
  Prefers gold_set_pooled.json (from Lab 01c); falls back to gold_set.json.
  Controlled by a single config constant — one-line change to swap.
"""
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
import os

# ---------------------------------------------------------------------------
# Config — single location to swap gold set paths
# ---------------------------------------------------------------------------
GOLD_SET_POOLED_PATH = r"C:\projects\Veritalent\retrieval_lab\evaluation\gold_set_pooled.json"
GOLD_SET_FALLBACK_PATH = r"C:\projects\Veritalent\retrieval_lab\evaluation\gold_set.json"

NDCG_K = 50
LOW_VALUE_THRESHOLD = 0.01  # Groups with |delta| < this flagged "low marginal value"


def get_gold_set_path() -> str:
    if os.path.exists(GOLD_SET_POOLED_PATH):
        return GOLD_SET_POOLED_PATH
    return GOLD_SET_FALLBACK_PATH


# ---------------------------------------------------------------------------
# Feature groups and weights
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "skill":     ["skill_depth", "skill_breadth", "skill_recency", "skill_mastery_triangulation"],
    "career":    ["career_velocity", "promotion_velocity", "tenure_stability", "inflection_point_strength"],
    "trust":     ["trust_score"],
    "activity":  ["activity_quality_composite"],
    "industry":  ["industry_relevance"],
    "logistics": ["logistics_fit_score"],
    "company":   ["product_vs_services"],
}

# Normalized weights summing to 1.0
GROUP_WEIGHTS = {
    "skill":     0.30,
    "career":    0.15,
    "activity":  0.20,
    "industry":  0.10,
    "company":   0.10,
    "trust":     0.10,
    "logistics": 0.05,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_feature_store(filepath: str) -> dict:
    candidates = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                candidates[row["candidate_id"]] = row
    return candidates


def normalize_features(candidates: dict) -> None:
    """Min-max normalize all feature values in-place, adding *_norm keys."""
    min_vals: dict = defaultdict(lambda: float("inf"))
    max_vals: dict = defaultdict(lambda: float("-inf"))

    for feats in candidates.values():
        for group_feats in FEATURE_GROUPS.values():
            for feat in group_feats:
                val = float(feats.get(feat, 0.0) or 0.0)
                if val < min_vals[feat]:
                    min_vals[feat] = val
                if val > max_vals[feat]:
                    max_vals[feat] = val

    for feats in candidates.values():
        for group_feats in FEATURE_GROUPS.values():
            for feat in group_feats:
                val = float(feats.get(feat, 0.0) or 0.0)
                denom = max_vals[feat] - min_vals[feat]
                feats[f"{feat}_norm"] = (val - min_vals[feat]) / denom if denom > 0 else 0.0


def score_candidate(feats: dict, exclude_group: str | None = None) -> float:
    score = 0.0
    for group, group_feats in FEATURE_GROUPS.items():
        if group == exclude_group:
            continue
        group_sum = sum(feats.get(f"{f}_norm", 0.0) for f in group_feats)
        score += group_sum * GROUP_WEIGHTS.get(group, 0.0)
    return score


def compute_ndcg_at_k(ranked_ids: list, judgments: dict, k: int = NDCG_K) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        rel = judgments.get(cid, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    ideal_rels = sorted(judgments.values(), reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation(feature_store_path: str) -> tuple[dict, float]:
    print("Loading feature store...")
    candidates = load_feature_store(feature_store_path)
    print(f"  Loaded {len(candidates):,} candidates.")

    print("Normalizing features...")
    normalize_features(candidates)

    gold_path = get_gold_set_path()
    print(f"Loading gold set from {gold_path}...")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    judgments = gold_data["queries"][0]["judgments"]
    print(f"  Gold set: {len(judgments)} candidates, "
          f"max relevance: {max(judgments.values())}")

    results = {}

    # Baseline
    baseline_scores = [(cid, score_candidate(feats)) for cid, feats in candidates.items()]
    baseline_scores.sort(key=lambda x: x[1], reverse=True)
    baseline_ndcg = compute_ndcg_at_k([x[0] for x in baseline_scores], judgments)
    results["baseline"] = baseline_ndcg
    print(f"\nBaseline NDCG@{NDCG_K}: {baseline_ndcg:.4f}")

    # Per-group ablation
    print("\n--- Ablation results ---")
    for group in FEATURE_GROUPS:
        scores = [(cid, score_candidate(feats, exclude_group=group))
                  for cid, feats in candidates.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        ndcg = compute_ndcg_at_k([x[0] for x in scores], judgments)
        delta = ndcg - baseline_ndcg
        flag = " ← LOW MARGINAL VALUE" if abs(delta) < LOW_VALUE_THRESHOLD else ""
        print(f"  Without {group:10s}: NDCG@{NDCG_K}={ndcg:.4f}  Δ={delta:+.4f}{flag}")
        results[group] = {"ndcg": ndcg, "delta": delta}

    return results, baseline_ndcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature group ablation testing")
    parser.add_argument("--store", required=True, help="Path to feature_store.jsonl")
    args = parser.parse_args()
    run_ablation(args.store)
