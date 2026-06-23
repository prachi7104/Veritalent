import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.checks.yoe_tenure_consistency import check_yoe_consistency
from trust_lab.checks.proficiency_plausibility import check_proficiency_plausibility

def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if the candidate matches either of the two documented honeypot patterns:
    1. |years_of_experience - (sum of career_history[].duration_months)/12| > 1.5
    2. 3+ skills rated "expert" with 0 months duration.
    """
    yoe_res = check_yoe_consistency(candidate)
    prof_res = check_proficiency_plausibility(candidate)
    
    return yoe_res["flagged"] or prof_res["flagged_honeypot_pattern"]

def get_known_honeypots(dataset_path: str) -> list:
    """
    Scans the dataset and returns a list of candidate dictionaries that match
    the known honeypot patterns.
    """
    honeypots = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if is_honeypot(cand):
                honeypots.append(cand)
    return honeypots

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"c:\projects\Veritalent\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    pots = get_known_honeypots(path)
    print(f"Found {len(pots)} known honeypots.")
