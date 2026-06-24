from typing import Dict, Any
import time
from retrieval_lab.evaluation.metrics import evaluate_run
from retrieval_lab.fusion.rrf import reciprocal_rank_fusion

def run(query: str, required_skills: list, judgments: Dict[str, int], bm25_index, dense_index, skill_graph, candidates_data, **kwargs) -> Dict[str, Any]:
    start_time = time.time()
    bm25_res = bm25_index.search(query, top_k=200)
    dense_res = dense_index.search(query, top_k=200)
    
    # Layer graph scores
    graph_scores = {}
    for cand_id in set([c["candidate_id"] for c in bm25_res] + [c["candidate_id"] for c in dense_res]):
        cand_dict = candidates_data[cand_id]
        score = skill_graph.score_candidate(cand_dict, required_skills)
        if score > 0:
            graph_scores[cand_id] = score
            
    # Sort graph results
    sorted_graph = [{"candidate_id": k, "score": v} for k, v in sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)]
    
    results = reciprocal_rank_fusion([bm25_res, dense_res, sorted_graph])
    latency_ms = (time.time() - start_time) * 1000
    
    metrics = evaluate_run(results, judgments)
    metrics["Latency (ms)"] = latency_ms
    metrics["Failure Note"] = "Skill adjacency can pull in false positives if the co-occurrence is coincidental."
    
    return metrics
