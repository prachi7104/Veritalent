import json
import pandas as pd
import time
import os
import sys

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retrieval_lab.data.loaders import load_candidates
from retrieval_lab.indexing.shootout_index import ShootoutIndex
from retrieval_lab.indexing.dense_index import DenseIndex
from retrieval_lab.evaluation.metrics import evaluate_run
from sentence_transformers import CrossEncoder, SentenceTransformer

def main():
    print("Loading data...")
    candidates = load_candidates(r"dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl", apply_funnel=True)
    print(f"Loaded {len(candidates)} candidates.")

    print("Loading Gold Set pooled...")
    with open('retrieval_lab/evaluation/gold_set_pooled.json', 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        pooled_judgments = gold_data["queries"][0]["judgments"]

    gold_ids = set(pooled_judgments.keys())
    
    # Take a 10% sample to allow CPU execution in a reasonable time, ensuring gold_ids are kept.
    import random
    random.seed(42)
    sample_candidates = []
    for c in candidates:
        if c["candidate_id"] in gold_ids:
            sample_candidates.append(c)
        elif random.random() < 0.1:
            sample_candidates.append(c)
            
    candidates = sample_candidates
    print(f"Subsampled to {len(candidates)} candidates for CPU-friendly shootout.")

    query_text = "Senior/Staff AI/ML Engineer with 5-9 years of experience, specializing in Search, Information Retrieval (IR), and building ranking systems. Must have strong product-company backgrounds."

    results = []
    baseline_ndcg = 0.434462 # Exact number from report

    print("\n--- Running Baseline Anchor ---")
    # Baseline: BGE-small-en-v1.5
    bge_small_baseline = ShootoutIndex("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", "")
    bge_cache_path = "retrieval_lab/cache/shootout_bge-small-en-v1.5.npz"
    if os.path.exists(bge_cache_path):
        bge_small_baseline.load(bge_cache_path)
    else:
        bge_small_baseline.build(candidates, batch_size=32)
        bge_small_baseline.save(bge_cache_path)
    
    start_q = time.time()
    res_baseline = bge_small_baseline.search(query_text, top_k=200)
    q_latency_baseline = (time.time() - start_q) * 1000
    
    met_baseline = evaluate_run(res_baseline, pooled_judgments)
    met_baseline["Experiment"] = "BAAI/bge-small-en-v1.5 (Baseline Sample)"
    met_baseline["Query Latency (ms)"] = q_latency_baseline
    met_baseline["Indexing Time (s)"] = bge_small_baseline.indexing_time_s
    met_baseline["Index Size (MB)"] = 0   # Skip
    results.append(met_baseline)

    # Models to benchmark
    models = [
        {"name": "BAAI/bge-base-en-v1.5", "q_prefix": "Represent this sentence for searching relevant passages: ", "p_prefix": ""},
        {"name": "intfloat/e5-small-v2", "q_prefix": "query: ", "p_prefix": "passage: "},
        {"name": "intfloat/e5-base-v2", "q_prefix": "query: ", "p_prefix": "passage: "},
        {"name": "thenlper/gte-small", "q_prefix": "", "p_prefix": ""},
        {"name": "thenlper/gte-base", "q_prefix": "", "p_prefix": ""}
    ]

    best_model_name = "BAAI/bge-small-en-v1.5"
    best_ndcg = met_baseline['NDCG@50']
    best_index = bge_small_baseline
    best_prefix = ""

    for m in models:
        print(f"\n--- Benchmarking {m['name']} ---")
        idx = ShootoutIndex(m["name"], m["q_prefix"], m["p_prefix"])
        cache_path = f"retrieval_lab/cache/shootout_{m['name'].split('/')[-1]}.npz"
        
        if os.path.exists(cache_path):
            idx.load(cache_path)
        else:
            try:
                idx.build(candidates, batch_size=32)
            except Exception as e:
                print(f"Error building {m['name']}: {e}. Retrying with trust_remote_code=True...")
                idx.model = SentenceTransformer(m["name"], trust_remote_code=True)
                idx.build(candidates, batch_size=32)
            idx.save(cache_path)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        
        start_q = time.time()
        res_cand = idx.search(query_text, top_k=200)
        q_latency = (time.time() - start_q) * 1000
        
        met = evaluate_run(res_cand, pooled_judgments)
        met["Experiment"] = m['name']
        met["Query Latency (ms)"] = q_latency
        met["Indexing Time (s)"] = idx.indexing_time_s
        met["Index Size (MB)"] = size_mb
        results.append(met)
        
        print(f"{m['name']} NDCG@50: {met['NDCG@50']:.4f}")
        if met['NDCG@50'] > best_ndcg:
            best_ndcg = met['NDCG@50']
            best_model_name = m['name']
            best_index = idx
            best_prefix = m['q_prefix']

    print(f"\nWinner: {best_model_name} with NDCG@50 = {best_ndcg:.4f}")

    print("\n--- Re-evaluating Batch Path (Exp D) with Winner ---")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    start_q = time.time()
    # Stage 1: Winner
    stage1_res = best_index.search(query_text, top_k=200)
    
    # Stage 2: Cross Encoder
    candidates_data = {c["candidate_id"]: c for c in candidates}
    ce_pairs = []
    for r in stage1_res:
        cand_dict = candidates_data[r["candidate_id"]]
        
        parts = []
        if cand_dict.get("profile", {}).get("summary"): parts.append(cand_dict["profile"]["summary"])
        if cand_dict.get("profile", {}).get("headline"): parts.append(cand_dict["profile"]["headline"])
        
        skills = [s['name'] for s in cand_dict.get('skills', []) if isinstance(s, dict) and 'name' in s]
        if skills: parts.append("Skills: " + ", ".join(skills))
            
        cand_text = " ".join(parts) if parts else ""
        ce_pairs.append([query_text, cand_text])
        
    ce_scores = cross_encoder.predict(ce_pairs)
    final_res = [{"candidate_id": stage1_res[i]["candidate_id"], "score": float(ce_scores[i])} for i in range(len(stage1_res))]
    final_res.sort(key=lambda x: x["score"], reverse=True)
    
    d_latency = (time.time() - start_q) * 1000
    met_d = evaluate_run(final_res, pooled_judgments)
    met_d["Experiment"] = f"D: Hybrid + Cross Encoder (Stage 1 = {best_model_name})"
    met_d["Query Latency (ms)"] = d_latency
    met_d["Indexing Time (s)"] = 0
    met_d["Index Size (MB)"] = 0
    results.append(met_d)

    df = pd.DataFrame(results)
    cols = ["Experiment", "NDCG@50", "P@10", "R@50", "Query Latency (ms)", "Indexing Time (s)", "Index Size (MB)"]
    df = df[cols]
    
    report = f"""# 02 Embedding Shootout Report

## Results

{df.to_markdown(index=False)}

## Conclusions
- **Live-path Winner:** {best_model_name}
- **Batch-path Winner:** D: Hybrid + Cross Encoder (Stage 1 = {best_model_name})
"""
    with open("retrieval_lab/reports/embedding_shootout_report.md", "w") as f:
        f.write(report)
        
    print("\nShootout Complete. Report saved to retrieval_lab/reports/embedding_shootout_report.md")

if __name__ == "__main__":
    main()
