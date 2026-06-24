import sys
import os
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from retrieval_lab.data.loaders import ALLOWED_TITLES
from trust_lab.evaluation.known_honeypot_set import is_honeypot

FINGERPRINT_SKILLS = set([
    "Search Backend", "Ranking Systems", "Text Encoders", "Vector Representations", 
    "Content Matching", "Model Adaptation", "Information Retrieval Systems", 
    "Search & Discovery", "Search Infrastructure", "Indexing Algorithms", 
    "Workflow Orchestration", "Open-source ML libraries", "Document Processing", 
    "Natural Language Processing"
])

def has_fingerprint(candidate: dict) -> bool:
    for skill in candidate.get("skills", []):
        if skill.get("name") in FINGERPRINT_SKILLS:
            return True
    return False

def sample_candidates(dataset_path: str, target_size: int = 250) -> list:
    """
    Selects ~250 candidates from the 31K allowlisted pool.
    Explicitly samples known honeypots, fingerprint holders, and a mix of activity signals.
    """
    allowlisted = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            title = cand.get("profile", {}).get("current_title", "").lower()
            if title in ALLOWED_TITLES:
                allowlisted.append(cand)
                
    honeypots = []
    fingerprints = []
    high_activity = []
    low_activity = []
    others = []
    
    for cand in allowlisted:
        if is_honeypot(cand):
            honeypots.append(cand)
        elif has_fingerprint(cand):
            fingerprints.append(cand)
        else:
            gh = cand.get("redrob_signals", {}).get("github_activity_score", -1)
            rr = cand.get("redrob_signals", {}).get("recruiter_response_rate", -1)
            if gh > 50 or rr > 0.8:
                high_activity.append(cand)
            elif (0 <= gh < 10) or (0 <= rr < 0.2):
                low_activity.append(cand)
            else:
                others.append(cand)
                
    sampled = []
    
    # Take all honeypots and fingerprints that are in the allowlist
    sampled.extend(honeypots)
    sampled.extend(fingerprints)
    
    # Take ~50 high activity, ~50 low activity
    sampled.extend(random.sample(high_activity, min(50, len(high_activity))))
    sampled.extend(random.sample(low_activity, min(50, len(low_activity))))
    
    # Fill remainder
    rem = target_size - len(sampled)
    if rem > 0:
        sampled.extend(random.sample(others, min(rem, len(others))))
        
    random.shuffle(sampled)
    return sampled[:target_size]

if __name__ == "__main__":
    sampled = sample_candidates(r"c:\projects\Veritalent\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl")
    out_path = r"c:\projects\Veritalent\ranking_lab\labels\stratified_sample.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2)
    print(f"Sampled {len(sampled)} candidates to {out_path}")
