# scripts/csv_diff.py
import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

def load_feature_store() -> dict:
    # Try loading from feature_store.jsonl or feature_store_v2.jsonl
    path_v2 = Path("feature_lab/store/feature_store_v2.jsonl")
    path_full = Path("feature_lab/store/feature_store.jsonl")
    store_path = path_full if (path_full.exists() and (not path_v2.exists() or path_full.stat().st_size > path_v2.stat().st_size)) else path_v2
    
    if not store_path.exists():
        print(f"[Warning] Feature store not found at {store_path}. Feature comparison will be skipped.")
        return {}
        
    store = {}
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            store[row["candidate_id"]] = row
    return store

def validate_format(df: pd.DataFrame, name: str) -> bool:
    print(f"\n--- Checking format of {name} ---")
    valid = True
    
    # Column check
    required_cols = ["rank", "candidate_id", "score", "reasoning"]
    if list(df.columns) != required_cols:
        print(f"[FAIL] Columns mismatch! Found: {list(df.columns)}. Expected: {required_cols}")
        valid = False
    else:
        print("[PASS] Required columns exist in order.")
        
    # Uniqueness check
    dups = df["candidate_id"].duplicated().sum()
    if dups > 0:
        print(f"[FAIL] Duplicate candidate_id entries found: {dups}")
        valid = False
    else:
        print("[PASS] No duplicate candidate_id entries.")
        
    # Ranks sequential check
    ranks = df["rank"].tolist()
    expected_ranks = list(range(1, len(df) + 1))
    if ranks != expected_ranks:
        print("[FAIL] Ranks are not sequential from 1 to N.")
        valid = False
    else:
        print("[PASS] Ranks are sequential from 1 to N.")
        
    # Scores sorting check
    scores = df["score"].tolist()
    is_sorted_desc = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    if not is_sorted_desc:
        print("[FAIL] Scores are not sorted in descending order!")
        valid = False
    else:
        print("[PASS] Scores are sorted in descending order.")
        
    return valid

def compare_csvs(baseline_path: str, new_path: str):
    print(f"Baseline: {baseline_path}")
    print(f"New:      {new_path}")
    
    df_base = pd.read_csv(baseline_path)
    df_new = pd.read_csv(new_path)
    
    b_ok = validate_format(df_base, "Baseline")
    n_ok = validate_format(df_new, "New Submission")
    
    if not b_ok or not n_ok:
        print("\n[WARNING] One or both files have validation errors! Comparison may be skewed.")
        
    # 1. Size Comparison
    print(f"\nSize Comparison:")
    print(f"  Baseline rows: {len(df_base)}")
    print(f"  New rows:      {len(df_new)}")
    
    # Find overlapping candidates
    base_ids = df_base["candidate_id"].tolist()
    new_ids = df_new["candidate_id"].tolist()
    intersection = set(base_ids) & set(new_ids)
    print(f"  Candidates in both: {len(intersection)} / {max(len(base_ids), len(new_ids))}")
    
    if not intersection:
        print("[FAIL] No common candidates between the files. Cannot perform rank comparison.")
        return
        
    # Map candidate_id to rank (0-indexed rank index)
    base_rank_map = {cid: idx for idx, cid in enumerate(base_ids)}
    new_rank_map = {cid: idx for idx, cid in enumerate(new_ids)}
    
    # 2. Correlation
    common_candidates = list(intersection)
    r_base = [base_rank_map[cid] for cid in common_candidates]
    r_new = [new_rank_map[cid] for cid in common_candidates]
    
    spearman_r, _ = scipy_stats.spearmanr(r_base, r_new)
    kendall_tau, _ = scipy_stats.kendalltau(r_base, r_new)
    
    print(f"\nRank Correlation (on overlapping candidates):")
    print(f"  Spearman r:  {spearman_r:.4f}")
    print(f"  Kendall Tau: {kendall_tau:.4f}")
    
    # 3. Top-10 Shifts
    top10_base = set(base_ids[:10])
    top10_new = set(new_ids[:10])
    entered = top10_new - top10_base
    exited = top10_base - top10_new
    
    print(f"\nTop-10 Shifts:")
    print(f"  Retained in Top-10: {len(top10_base & top10_new)} / 10")
    if entered:
        print(f"  Entered Top-10:     {', '.join(entered)}")
        for cid in entered:
            print(f"    - {cid} was previously rank {base_rank_map.get(cid, 'N/A') + 1 if cid in base_rank_map else 'not in list'}")
    else:
        print("  No new entries in Top-10.")
    if exited:
        print(f"  Exited Top-10:      {', '.join(exited)}")
        for cid in exited:
            print(f"    - {cid} is now rank {new_rank_map.get(cid, 'N/A') + 1 if cid in new_rank_map else 'not in list'}")
            
    # 4. Significant Rank Shifts
    shifts = []
    for cid in intersection:
        shift = base_rank_map[cid] - new_rank_map[cid]  # positive if moved UP (closer to rank 1)
        shifts.append((cid, base_rank_map[cid] + 1, new_rank_map[cid] + 1, shift))
        
    shifts.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"\nTop Rank Changes (Significant shifts):")
    printed = 0
    for cid, old_r, new_r, shift in shifts:
        if abs(shift) >= 5:
            direction = "UP" if shift > 0 else "DOWN"
            print(f"  {cid}: Rank {old_r} → {new_r} ({direction} by {abs(shift)})")
            printed += 1
            if printed >= 10:
                break
    if printed == 0:
        print("  No candidates shifted by 5 or more positions.")
        
    # 5. Feature Distribution Comparison
    store = load_feature_store()
    if store:
        feats_to_check = [
            "skill_depth", "skill_breadth", "jd_skill_score",
            "tenure_stability", "activity_quality_composite", "trust_score"
        ]
        print(f"\nFeature Distribution Comparison (Top 25 averages):")
        print(f"  {'Feature':<30} | {'Baseline Avg':<15} | {'New Avg':<15} | {'Delta':<10}")
        print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")
        
        base_top25 = base_ids[:25]
        new_top25 = new_ids[:25]
        
        for feat in feats_to_check:
            base_vals = [float(store[cid].get(feat, 0.0) or 0.0) for cid in base_top25 if cid in store]
            new_vals = [float(store[cid].get(feat, 0.0) or 0.0) for cid in new_top25 if cid in store]
            
            avg_base = np.mean(base_vals) if base_vals else 0.0
            avg_new = np.mean(new_vals) if new_vals else 0.0
            delta = avg_new - avg_base
            
            print(f"  {feat:<30} | {avg_base:<15.4f} | {avg_new:<15.4f} | {delta:<+10.4f}")
            
        # Trust (risk) impact check details
        print(f"\nTrust Risk Check (Top 25):")
        base_trust = [float(store[cid].get("trust_score", 0.0) or 0.0) for cid in base_top25 if cid in store]
        new_trust = [float(store[cid].get("trust_score", 0.0) or 0.0) for cid in new_top25 if cid in store]
        
        base_high_risk = sum(1 for v in base_trust if v > 0.5)
        new_high_risk = sum(1 for v in new_trust if v > 0.5)
        print(f"  High risk candidates (trust_score > 0.5) in top-25:")
        print(f"    Baseline: {base_high_risk} / {len(base_trust)}")
        print(f"    New:      {new_high_risk} / {len(new_trust)}")

def main():
    parser = argparse.ArgumentParser(description="CSV Diff & Validation Tool")
    parser.add_argument("baseline", help="Path to baseline CSV")
    parser.add_argument("new", help="Path to new submission CSV")
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline):
        print(f"[ERROR] Baseline CSV not found at {args.baseline}")
        sys.exit(1)
    if not os.path.exists(args.new):
        print(f"[ERROR] New CSV not found at {args.new}")
        sys.exit(1)
        
    compare_csvs(args.baseline, args.new)

if __name__ == "__main__":
    main()
