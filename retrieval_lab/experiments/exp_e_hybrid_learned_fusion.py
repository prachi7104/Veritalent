from typing import Dict, Any
import time
from retrieval_lab.evaluation.metrics import evaluate_run

def run(query: str, judgments: Dict[str, int], bm25_index, dense_index, learned_fusion, **kwargs) -> Dict[str, Any]:
    start_time = time.time()
    bm25_res = bm25_index.search(query, top_k=200)
    dense_res = dense_index.search(query, top_k=200)
    
    bm25_scores = {res["candidate_id"]: res["score"] for res in bm25_res}
    dense_scores = {res["candidate_id"]: res["score"] for res in dense_res}
    
    results = learned_fusion.score(bm25_scores, dense_scores)
    latency_ms = (time.time() - start_time) * 1000
    
    metrics = evaluate_run(results, judgments)
    metrics["Latency (ms)"] = latency_ms
    metrics["Failure Note"] = "Learned fusion is sensitive to the scale of raw scores and might overfit the gold set."
    
    return metrics
