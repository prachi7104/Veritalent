import numpy as np
from typing import List, Dict, Any

def precision_at_k(ranked_ids: List[str], gold_judgments: Dict[str, int], k: int, threshold: int = 2) -> float:
    """
    Precision at k. A candidate is considered relevant if their gold score >= threshold.
    """
    if k == 0 or not ranked_ids:
        return 0.0
        
    top_k = ranked_ids[:k]
    relevant_count = 0
    for cand_id in top_k:
        if gold_judgments.get(cand_id, 0) >= threshold:
            relevant_count += 1
            
    return relevant_count / len(top_k)

def recall_at_k(ranked_ids: List[str], gold_judgments: Dict[str, int], k: int, threshold: int = 2) -> float:
    """
    Recall at k.
    """
    total_relevant = sum(1 for score in gold_judgments.values() if score >= threshold)
    if total_relevant == 0:
        return 0.0
        
    top_k = ranked_ids[:k]
    relevant_count = 0
    for cand_id in top_k:
        if gold_judgments.get(cand_id, 0) >= threshold:
            relevant_count += 1
            
    return relevant_count / total_relevant

def ndcg_at_k(ranked_ids: List[str], gold_judgments: Dict[str, int], k: int) -> float:
    """
    NDCG at k. Uses raw relevance scores (0 to 3).
    """
    if k == 0 or not ranked_ids:
        return 0.0
        
    top_k = ranked_ids[:k]
    dcg = 0.0
    for i, cand_id in enumerate(top_k):
        rel = gold_judgments.get(cand_id, 0)
        dcg += (2**rel - 1) / np.log2(i + 2)
        
    # Ideal DCG
    ideal_scores = sorted([score for score in gold_judgments.values()], reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        idcg += (2**rel - 1) / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def evaluate_run(ranked_results: List[Dict[str, Any]], gold_judgments: Dict[str, int]) -> Dict[str, float]:
    """
    Returns P@10, P@50, R@50, NDCG@50
    """
    ranked_ids = [res["candidate_id"] for res in ranked_results]
    
    return {
        "P@10": precision_at_k(ranked_ids, gold_judgments, 10),
        "P@50": precision_at_k(ranked_ids, gold_judgments, 50),
        "R@50": recall_at_k(ranked_ids, gold_judgments, 50),
        "NDCG@50": ndcg_at_k(ranked_ids, gold_judgments, 50)
    }
