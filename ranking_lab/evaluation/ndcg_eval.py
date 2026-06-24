"""
ranking_lab/evaluation/ndcg_eval.py

Core NDCG@K evaluation utilities. All experiments evaluate against the 
gold_set_pooled.json judgments at K=10 and K=50.
"""
import json
import numpy as np
from pathlib import Path

GOLD_SET_PATH = Path(__file__).parent.parent.parent / "retrieval_lab" / "evaluation" / "gold_set_pooled.json"


def load_gold_set(path: str = None) -> dict:
    p = Path(path) if path else GOLD_SET_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # Return dict: {candidate_id: relevance_label}
    judgments = {}
    for query in data["queries"]:
        judgments.update(query["judgments"])
    return judgments


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances)]
    return sum(gains)


def ndcg_at_k(ranked_candidate_ids: list[str], gold_judgments: dict, k: int = 10) -> float:
    """
    Computes NDCG@K given:
      - ranked_candidate_ids: ordered list of candidate IDs (best first)
      - gold_judgments: dict mapping candidate_id -> relevance label (0-3)
      - k: cutoff
    """
    relevances = [gold_judgments.get(cid, 0) for cid in ranked_candidate_ids]
    actual_dcg = dcg_at_k(relevances, k)

    # Ideal DCG: sort all known relevant items by label desc
    ideal_relevances = sorted(gold_judgments.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(ranked_candidate_ids: list[str], gold_judgments: dict, k: int = 10, threshold: int = 1) -> float:
    """P@K: fraction of top-K that are relevant (label >= threshold)."""
    top_k = ranked_candidate_ids[:k]
    relevant = sum(1 for cid in top_k if gold_judgments.get(cid, 0) >= threshold)
    return relevant / k if k > 0 else 0.0


def recall_at_k(ranked_candidate_ids: list[str], gold_judgments: dict, k: int = 50, threshold: int = 1) -> float:
    """R@K: fraction of all relevant candidates found in top-K."""
    top_k = set(ranked_candidate_ids[:k])
    all_relevant = {cid for cid, lbl in gold_judgments.items() if lbl >= threshold}
    if not all_relevant:
        return 0.0
    return len(top_k & all_relevant) / len(all_relevant)


def evaluate_ranking(ranked_candidate_ids: list[str], gold_judgments: dict) -> dict:
    """Run full evaluation suite and return metrics dict."""
    return {
        "ndcg@10": ndcg_at_k(ranked_candidate_ids, gold_judgments, k=10),
        "ndcg@50": ndcg_at_k(ranked_candidate_ids, gold_judgments, k=50),
        "p@10": precision_at_k(ranked_candidate_ids, gold_judgments, k=10),
        "p@50": precision_at_k(ranked_candidate_ids, gold_judgments, k=50),
        "r@50": recall_at_k(ranked_candidate_ids, gold_judgments, k=50),
    }
