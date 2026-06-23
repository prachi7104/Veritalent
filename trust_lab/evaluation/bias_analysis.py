import os
import sys
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.scoring.ensemble_trust_score import compute_trust_score

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m")
        except ValueError:
            return None

def compute_career_gaps(career_history):
    # Sort history by start date
    valid_entries = []
    for entry in career_history:
        start = parse_date(entry.get("start_date"))
        end = parse_date(entry.get("end_date"))
        if start:
            valid_entries.append((start, end))
            
    valid_entries.sort(key=lambda x: x[0])
    gaps = 0
    
    for i in range(1, len(valid_entries)):
        prev_end = valid_entries[i-1][1]
        curr_start = valid_entries[i][0]
        
        if prev_end and curr_start:
            # If gap > 90 days, count it as a gap
            if (curr_start - prev_end).days > 90:
                gaps += 1
    return gaps

def compute_industry_switches(career_history):
    valid_entries = []
    for entry in career_history:
        start = parse_date(entry.get("start_date"))
        if start:
            valid_entries.append((start, entry.get("industry")))
            
    valid_entries.sort(key=lambda x: x[0])
    switches = 0
    for i in range(1, len(valid_entries)):
        prev_ind = valid_entries[i-1][1]
        curr_ind = valid_entries[i][1]
        if prev_ind and curr_ind and prev_ind != curr_ind:
            switches += 1
    return switches

def is_technical_study(education):
    # simple heuristic
    tech_keywords = ['computer', 'software', 'engineering', 'science', 'mathematics', 'physics', 'data', 'information', 'technology']
    for edu in education:
        field = edu.get("field_of_study", "").lower()
        if any(k in field for k in tech_keywords):
            return True
    return False

def run_bias_analysis(dataset_path: str, threshold: float = 0.4):
    gaps_list = []
    switches_list = []
    tech_list = []
    flagged_list = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            
            career_history = cand.get("career_history", [])
            gaps = compute_career_gaps(career_history)
            switches = compute_industry_switches(career_history)
            tech = is_technical_study(cand.get("education", []))
            
            score_res = compute_trust_score(cand)
            flagged = 1 if score_res["trust_score"] >= threshold else 0
            
            gaps_list.append(gaps)
            switches_list.append(switches)
            tech_list.append(1 if tech else 0)
            flagged_list.append(flagged)
            
    # Compute point-biserial correlations (or just Pearson since both are numeric)
    # Check if arrays are constant to avoid NaNs
    if np.std(gaps_list) == 0 or np.std(flagged_list) == 0:
        gaps_corr = 0.0
    else:
        gaps_corr = np.corrcoef(gaps_list, flagged_list)[0, 1]
        
    if np.std(switches_list) == 0 or np.std(flagged_list) == 0:
        switches_corr = 0.0
    else:
        switches_corr = np.corrcoef(switches_list, flagged_list)[0, 1]
        
    if np.std(tech_list) == 0 or np.std(flagged_list) == 0:
        tech_corr = 0.0
    else:
        tech_corr = np.corrcoef(tech_list, flagged_list)[0, 1]
    
    print("--- Bias Analysis Results ---")
    print(f"Correlation with Career Gaps: {gaps_corr:.3f}")
    print(f"Correlation with Industry Switches: {switches_corr:.3f}")
    print(f"Correlation with Technical Degree: {tech_corr:.3f}")
    
    print("\nMitigation Recommendation:")
    print("If correlations are > 0.1, it suggests bias against non-linear paths.")
    print("Applied Mitigation: YOE check must use a minimum deviation threshold (1.5 years) ")
    print("and a scaled penalty that tolerates typical gap patterns before reaching maximum risk.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"c:\projects\Veritalent\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    run_bias_analysis(path)
