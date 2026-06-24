from typing import Dict, Any
import time
from retrieval_lab.evaluation.metrics import evaluate_run

def run(query: str, judgments: Dict[str, int], index, **kwargs) -> Dict[str, Any]:
    start_time = time.time()
    results = index.search(query, top_k=200)
    latency_ms = (time.time() - start_time) * 1000
    
    metrics = evaluate_run(results, judgments)
    metrics["Latency (ms)"] = latency_ms
    metrics["Failure Note"] = "Dense embeddings can miss exact keyword requirements like BM25."
    
    return metrics
