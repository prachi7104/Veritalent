from typing import Dict, Any
import time
from retrieval_lab.evaluation.metrics import evaluate_run
from retrieval_lab.fusion.rrf import reciprocal_rank_fusion

def run(query: str, judgments: Dict[str, int], bm25_index, dense_index, cross_encoder, candidates_data, **kwargs) -> Dict[str, Any]:
    start_time = time.time()
    bm25_res = bm25_index.search(query, top_k=200)
    dense_res = dense_index.search(query, top_k=200)
    fused = reciprocal_rank_fusion([bm25_res, dense_res])
    
    reranked, ce_latency = cross_encoder.rerank(query, fused, candidates_data, top_k=200)
    latency_ms = ((time.time() - start_time) * 1000)
    
    metrics = evaluate_run(reranked, judgments)
    metrics["Latency (ms)"] = latency_ms
    metrics["Failure Note"] = "High latency due to cross-encoder inference. Limited to top 200, so recall cannot exceed stage 1."
    
    return metrics
