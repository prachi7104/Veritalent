from typing import Dict, Any
import time
from retrieval_lab.evaluation.metrics import evaluate_run
from retrieval_lab.fusion.rrf import reciprocal_rank_fusion

def run(query: str, query_parts: Dict[str, str], judgments: Dict[str, int], bm25_index, multivector_index, **kwargs) -> Dict[str, Any]:
    start_time = time.time()
    bm25_res = bm25_index.search(query, top_k=200)
    mv_res = multivector_index.search(query_parts, weights={"summary": 1.0, "headline": 0.5, "skills": 1.5}, top_k=200)
    
    results = reciprocal_rank_fusion([bm25_res, mv_res])
    latency_ms = (time.time() - start_time) * 1000
    
    metrics = evaluate_run(results, judgments)
    metrics["Latency (ms)"] = latency_ms
    metrics["Failure Note"] = "Requires structured query parsing; performance degrades if query mapping to fields is poor."
    
    return metrics
