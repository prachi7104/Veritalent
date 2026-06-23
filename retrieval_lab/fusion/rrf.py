from typing import List, Dict, Any

def reciprocal_rank_fusion(lists_of_candidates: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Implements Reciprocal Rank Fusion (RRF).
    lists_of_candidates is a list where each element is a list of candidate dictionaries containing at least 'candidate_id'.
    k is a constant that mitigates the impact of high rankings by outlier systems.
    """
    rrf_scores = {}
    
    for candidate_list in lists_of_candidates:
        if not candidate_list:
            continue
            
        # Normalize raw scores to [0, 1] for tie-breaking
        raw_scores = [cand["score"] for cand in candidate_list]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        range_score = max_score - min_score if max_score > min_score else 1.0
        
        for rank, cand in enumerate(candidate_list):
            cand_id = cand["candidate_id"]
            if cand_id not in rrf_scores:
                rrf_scores[cand_id] = 0.0
            
            # Base RRF score
            base_rrf = 1.0 / (k + rank + 1)
            
            # Tie-breaker using normalized original score (scaled down so it never overrides base RRF rank tiers)
            normalized_raw = (cand["score"] - min_score) / range_score
            tie_breaker = normalized_raw * 1e-6
            
            rrf_scores[cand_id] += base_rrf + tie_breaker
            
    # Sort by RRF score descending
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for cand_id, score in sorted_candidates:
        results.append({
            "candidate_id": cand_id,
            "score": score
        })
        
    return results
