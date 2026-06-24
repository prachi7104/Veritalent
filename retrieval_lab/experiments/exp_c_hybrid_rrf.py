from typing import Dict, Any
import time
from retrieval_lab.evaluation.metrics import evaluate_run
from retrieval_lab.fusion.rrf import reciprocal_rank_fusion

def run(query: str, judgments: Dict[str, int], bm25_index, dense_index, **kwargs) -> Dict[str, Any]:
    start_time = time.time()
    bm25_res = bm25_index.search(query, top_k=200)
    dense_res = dense_index.search(query, top_k=200)
    
    results = reciprocal_rank_fusion([bm25_res, dense_res])
    latency_ms = (time.time() - start_time) * 1000
    
    metrics = evaluate_run(results, judgments)
    metrics["Latency (ms)"] = latency_ms
    metrics["Failure Note"] = "RRF is rank-based and ignores magnitude of scores, sometimes penalizing strong dense matches."
    
    return metrics
