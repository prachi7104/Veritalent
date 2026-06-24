import json
import random
from pathlib import Path

from .loaders import load_candidates_iter

DEEP_IR_SKILLS = {
    "PyTorch", "TensorFlow", "NLP", "Machine Learning", "Deep Learning", 
    "BM25", "Learning to Rank", "Qdrant", "Weaviate", "Milvus", 
    "scikit-learn", "Elasticsearch", "OpenSearch", "LlamaIndex", 
    "Haystack", "QLoRA", "PEFT", "LoRA", "pgvector", "Natural Language Processing"
}

def is_senior_ai_title(title: str) -> bool:
    title = title.lower()
    has_ai_ml = any(x in title for x in ["ai", "ml", "machine learning", "artificial intelligence", "data scientist"])
    is_senior = any(x in title for x in ["senior", "staff", "lead", "principal", "sr", "head"])
    return has_ai_ml and is_senior

def count_deep_ir_skills(candidate: dict) -> int:
    count = 0
    for skill in candidate.get("skills", []):
        if skill.get("name") in DEEP_IR_SKILLS:
            count += 1
    return count

def compute_heuristic_score(candidate: dict) -> int:
    """
    Heuristic scoring for Senior/Staff AI/ML Engineer - Search & Retrieval (Pune/Noida)
    0 = irrelevant, 1 = plausible, 2 = strong fit, 3 = excellent fit
    """
    title = candidate.get("profile", {}).get("current_title", "")
    location = candidate.get("profile", {}).get("location", "").lower()
    
    deep_ir_count = count_deep_ir_skills(candidate)
    senior_ai = is_senior_ai_title(title)
    in_target_loc = "pune" in location or "noida" in location
    
    summary = candidate.get("profile", {}).get("summary", "").lower()
    has_search_exp = "search" in summary or "retrieval" in summary
    
    if senior_ai and deep_ir_count >= 3 and in_target_loc:
        return 3
    elif (senior_ai and deep_ir_count >= 3) or (has_search_exp and deep_ir_count >= 4):
        return 2
    elif deep_ir_count >= 2 or (senior_ai and deep_ir_count >= 1):
        return 1
    else:
        return 0

def generate_gold_set(filepath: str, output_path: str, max_per_class: int = 25):
    """
    Generates a stratified gold set from the full candidate pool based on heuristic scoring.
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    gold_set = []
    
    print("Generating gold set...")
    for candidate in load_candidates_iter(filepath):
        score = compute_heuristic_score(candidate)
        if counts[score] < max_per_class:
            # We store only what's necessary for the gold set, typically candidate_id and score
            gold_set.append({
                "candidate_id": candidate["candidate_id"],
                "relevance_score": score,
                "heuristic_details": {
                    "title": candidate.get("profile", {}).get("current_title", ""),
                    "deep_ir_count": count_deep_ir_skills(candidate)
                }
            })
            counts[score] += 1
            
        if all(c >= max_per_class for c in counts.values()):
            break
            
    # Save to JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "metadata": {
            "description": "Heuristically constructed validation set for Senior/Staff AI/ML Engineer - Search & Retrieval",
            "scoring_logic": "0: Irrelevant, 1: Plausible (some IR skills), 2: Strong (Senior AI title + 3 IR skills), 3: Excellent (Strong + Pune/Noida location)",
            "limitations": "This is a synthetically generated set based on keyword and title heuristics, not human-verified ground truth."
        },
        "queries": [
            {
                "query_id": "q1",
                "text": "Senior/Staff AI/ML Engineer — Search & Retrieval. 5-9 years of experience, strong production NLP/IR skills, prefers product-company backgrounds, soft preference for sub-30-day notice periods. Down-weight inactive candidates.",
                "judgments": {
                    item["candidate_id"]: item["relevance_score"] for item in gold_set
                }
            }
        ]
    }
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Gold set generated with {len(gold_set)} candidates at {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sample_validation_set.py <input_candidates.jsonl> <output_gold_set.json>")
        sys.exit(1)
    generate_gold_set(sys.argv[1], sys.argv[2])
