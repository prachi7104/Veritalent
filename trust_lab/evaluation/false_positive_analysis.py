import os
import sys
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.scoring.ensemble_trust_score import compute_trust_score
from trust_lab.evaluation.known_honeypot_set import is_honeypot

def run_fp_analysis(dataset_path: str, sample_size: int = 10, threshold: float = 0.4):
    """
    Identifies false positives: candidates who get a high trust score (indicating risk)
    but do NOT match the rigid honeypot patterns.
    """
    false_positives = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            
            # Skip actual honeypots
            if is_honeypot(cand):
                continue
                
            score_res = compute_trust_score(cand)
            if score_res["trust_score"] >= threshold:
                false_positives.append((cand, score_res))
                
    print(f"Found {len(false_positives)} false positives (score >= {threshold}, not known honeypot).")
    
    # Sample and characterize
    if false_positives:
        sample = random.sample(false_positives, min(sample_size, len(false_positives)))
        print("\nSample False Positives for Characterization:")
        for cand, res in sample:
            print(f"\n--- Candidate: {cand.get('candidate_id')} ---")
            print(f"Score: {res['trust_score']:.3f}")
            print("Breakdown:", json.dumps(res['breakdown'], indent=2))
            print("Details:", json.dumps(res['details'], indent=2))
            
    return false_positives

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"c:\projects\Veritalent\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    run_fp_analysis(path)
