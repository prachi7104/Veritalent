import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Dict, List, Set
from collections import defaultdict
from retrieval_lab.data.loaders import load_candidates
from retrieval_lab.indexing.bm25_index import BM25Index
from retrieval_lab.indexing.dense_index import DenseIndex
from retrieval_lab.indexing.multivector_index import MultiVectorIndex
from retrieval_lab.indexing.skill_graph import SkillGraph

def score_candidate(cand: Dict) -> int:
    title = cand.get('profile', {}).get('current_title', '').lower()
    location = cand.get('profile', {}).get('location', '').lower()
    skills = [s['name'].lower() for s in cand.get('skills', [])]
    
    ir_skills = {
        'nlp', 'information retrieval', 'machine learning', 'deep learning', 'bm25',
        'search', 'elasticsearch', 'solr', 'learning to rank', 'recommendation systems',
        'recsys', 'natural language processing', 'langchain', 'sentence transformers',
        'weaviate', 'pinecone', 'milvus', 'qdrant', 'vector search', 'transformers', 'huggingface'
    }
    cand_ir_skills = [s for s in skills if any(ir in s for ir in ir_skills)]
    
    is_senior_ai = any(prefix in title for prefix in ['senior', 'staff', 'lead', 'principal', 'data scientist', 'ai', 'ml'])
    is_pune_noida = 'pune' in location or 'noida' in location
    
    if is_senior_ai and len(cand_ir_skills) >= 3:
        if is_pune_noida:
            return 3
        return 2
    if len(cand_ir_skills) > 0:
        return 1
    return 0

def build_pooled_gold_set():
    # 1. Load candidates (must use the allowlist filtered ones, or we can load all. Let's use allowlist since these are the valid ones)
    candidates_path = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    candidates = load_candidates(candidates_path, limit=None, apply_funnel=True)
    candidates_data = {c["candidate_id"]: c for c in candidates}
    
    # Load original gold set
    gold_set_path = "retrieval_lab/evaluation/gold_set.json"
    with open(gold_set_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    original_judgments = gold_data["queries"][0]["judgments"]
    query_text = gold_data["queries"][0]["text"]
    
    # Load indices
    cache_dir = "retrieval_lab/cache"
    print("Loading indices...")
    bm25 = BM25Index()
    bm25.load(os.path.join(cache_dir, "bm25.pkl"))
    
    dense = DenseIndex()
    dense.load(os.path.join(cache_dir, "dense.npz"))
    
    mv = MultiVectorIndex()
    mv.load(os.path.join(cache_dir, "multivector.npz"))
    
    sg = SkillGraph()
    sg.load(os.path.join(cache_dir, "skill_graph.pkl"))
    
    print("Running searches to collect top-50...")
    res_a = bm25.search(query_text, top_k=50)
    res_b = dense.search(query_text, top_k=50)
    
    # For Exp C (Hybrid RRF) - we need to run it. RRF combines top 200, so we do top 200 then take top 50
    from retrieval_lab.fusion.rrf import reciprocal_rank_fusion
    res_a_200 = bm25.search(query_text, top_k=200)
    res_b_200 = dense.search(query_text, top_k=200)
    res_c_all = reciprocal_rank_fusion([res_a_200, res_b_200])
    res_c = res_c_all[:50]
    
    # For Exp F (MultiVector RRF)
    query_parts = {
        "summary": "5-9 years of experience, product-company backgrounds",
        "headline": "Senior/Staff AI/ML Engineer Search Retrieval",
        "skills": "NLP IR Machine Learning Deep Learning BM25"
    }
    res_f_mv = mv.search(query_parts, top_k=200)
    res_f_all = reciprocal_rank_fusion([res_a_200, res_f_mv])
    res_f = res_f_all[:50]
    
    # For Exp G (Skill Graph Recall)
    required_skills = ["BM25", "Learning to Rank", "PyTorch", "NLP", "Elasticsearch"]
    graph_scores = {}
    for cand_id in set([c["candidate_id"] for c in res_a_200] + [c["candidate_id"] for c in res_b_200]):
        cand_dict = candidates_data[cand_id]
        score = sg.score_candidate(cand_dict, required_skills)
        if score > 0:
            graph_scores[cand_id] = score
            
    sorted_graph = [{"candidate_id": k, "score": v} for k, v in sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)]
    res_g_all = reciprocal_rank_fusion([res_a_200, res_b_200, sorted_graph])
    res_g = res_g_all[:50]
    
    # Collect candidate IDs
    pool = defaultdict(set)
    for res in res_a: pool[res['candidate_id']].add("A")
    for res in res_b: pool[res['candidate_id']].add("B")
    for res in res_c: pool[res['candidate_id']].add("C")
    for res in res_f: pool[res['candidate_id']].add("F")
    for res in res_g: pool[res['candidate_id']].add("G")
    
    # Also ensure original gold set candidates are in the pool
    for gid in original_judgments:
        pool[gid].add("ORIGINAL_GOLD")
    
    print(f"Total unique candidates in pool: {len(pool)}")
    
    # Score candidates
    pooled_judgments = {}
    pool_sources = {}
    
    new_positive_cases = 0
    semantic_only_new_positives = 0
    
    for gid, sources in pool.items():
        if gid in original_judgments:
            score = original_judgments[gid]
        else:
            if gid in candidates_data:
                score = score_candidate(candidates_data[gid])
            else:
                score = 0 # Drop/irrelevant if not in valid candidate pool
                
            if score >= 1:
                new_positive_cases += 1
                if "A" not in sources: # only in B, C, F, G
                    semantic_only_new_positives += 1
                    
        pooled_judgments[gid] = score
        pool_sources[gid] = sorted(list(sources))
        
    print(f"Original positive cases: {sum(1 for s in original_judgments.values() if s >= 1)}")
    print(f"New positive cases added: {new_positive_cases}")
    print(f"New positive cases from semantic methods ONLY (B/C/F/G): {semantic_only_new_positives}")
    
    # Build new gold data
    gold_data["metadata"]["description"] = "Pooled heuristic validation set (BM25 + Dense + Hybrid + MultiVector + SkillGraph)"
    gold_data["metadata"]["limitations"] = "Synthetically generated set based on pooling multiple retrieval outputs and scoring them via heuristics to eliminate single-method pooling bias."
    
    gold_data["queries"][0]["judgments"] = pooled_judgments
    gold_data["queries"][0]["pool_source"] = pool_sources
    
    out_path = "retrieval_lab/evaluation/gold_set_pooled.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=2)
        
    print(f"Pooled gold set saved to {out_path}")

if __name__ == "__main__":
    build_pooled_gold_set()
