# ranking_lab/evaluation/eval_metrics.py
"""
Shared evaluation metrics for all ranking experiments.
Every experiment MUST call these functions so results are comparable.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int) -> float:
    """
    Compute NDCG@k given a ranked list and a relevance dict {candidate_id: 0-3}.
    """
    def dcg(ids, rel, k):
        gains = []
        for i, cid in enumerate(ids[:k]):
            r = rel.get(cid, 0)
            gains.append((2**r - 1) / np.log2(i + 2))
        return sum(gains)

    ideal_order = sorted(relevance.keys(), key=lambda x: -relevance[x])
    idcg = dcg(ideal_order, relevance, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked_ids, relevance, k) / idcg


def precision_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int,
                   threshold: int = 2) -> float:
    """Precision@k: fraction of top-k candidates with relevance >= threshold."""
    hits = sum(1 for cid in ranked_ids[:k] if relevance.get(cid, 0) >= threshold)
    return hits / k


def recall_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int,
                threshold: int = 2) -> float:
    """Recall@k: fraction of all relevant candidates found in top-k."""
    total_relevant = sum(1 for r in relevance.values() if r >= threshold)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if relevance.get(cid, 0) >= threshold)
    return hits / total_relevant


def score_spread(scores: List[float]) -> Dict[str, float]:
    """Measure how discriminative the scores are — catch score clustering."""
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "n_ties": int(len(arr) - len(set(arr.round(4)))),
    }


def feature_dominance_check(shap_summaries: List[List[dict]]) -> Dict[str, int]:
    """
    Count how many candidates have each feature in their SHAP top-5.
    A feature appearing in 90%+ of candidates is a dominance red flag.
    """
    from collections import Counter
    counter = Counter()
    for summary in shap_summaries:
        for item in summary:
            counter[item["feature"]] += 1
    return dict(counter.most_common())


def full_eval_report(
    experiment_name: str,
    ranked_ids: List[str],
    scores: List[float],
    relevance: Dict[str, int],
    shap_summaries: List[List[dict]] = None,
    latency_ms: float = None,
) -> dict:
    """
    Produce a complete evaluation report dict.
    Save this to reports_archive/ after every experiment.
    """
    report = {
        "experiment": experiment_name,
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
        "ndcg@50": ndcg_at_k(ranked_ids, relevance, 50),
        "p@10": precision_at_k(ranked_ids, relevance, 10),
        "p@50": precision_at_k(ranked_ids, relevance, 50),
        "r@50": recall_at_k(ranked_ids, relevance, 50),
        "score_spread": score_spread(scores),
    }
    if shap_summaries:
        report["feature_dominance"] = feature_dominance_check(shap_summaries)
    if latency_ms is not None:
        report["latency_ms"] = latency_ms
    return report


def compare_to_baseline(new_report: dict, baseline_ndcg10: float = 0.7473) -> dict:
    """Compare experiment results against the known baseline."""
    delta = new_report["ndcg@10"] - baseline_ndcg10
    return {
        "baseline_ndcg@10": baseline_ndcg10,
        "new_ndcg@10": new_report["ndcg@10"],
        "delta": round(delta, 4),
        "verdict": "IMPROVEMENT" if delta > 0.005 else (
                   "REGRESSION" if delta < -0.005 else "NEUTRAL"),
    }
