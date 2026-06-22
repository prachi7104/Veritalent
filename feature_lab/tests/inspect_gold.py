import json
import sys
import os

sys.path.append(r'C:\projects\Veritalent')

# Import the features directly
from feature_lab.features.skill_features import SkillDepthFeature, SkillBreadthFeature
from feature_lab.features.activity_features import ActivityQualityCompositeFeature

# Load sample candidate from candidates.jsonl
cand_path = r"C:\projects\Veritalent\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

# Load gold set to find a high-relevance candidate ID
with open(r'C:\projects\Veritalent\retrieval_lab\evaluation\gold_set_pooled.json', 'r') as f:
    gs = json.load(f)

judgments = gs['queries'][0]['judgments']
high_rel_ids = {cid for cid, rel in judgments.items() if rel >= 2}

print(f"Looking for high-rel candidates: {list(high_rel_ids)[:5]}")

found = 0
with open(cand_path, 'r') as f:
    for line in f:
        row = json.loads(line)
        cid = row['candidate_id']
        if cid in high_rel_ids:
            print(f"\n=== {cid} (rel={judgments[cid]}) ===")
            profile = row.get('profile', {})
            print(f"  Title: {profile.get('current_title', 'N/A')}")
            print(f"  Skills: {[s['name'] for s in row.get('skills', [])]}")
            
            signals = row.get('redrob_signals', {})
            print(f"  redrob_signals keys: {list(signals.keys())}")
            print(f"  last_active_date: {signals.get('last_active_date')}")
            print(f"  github_activity_score: {signals.get('github_activity_score')}")
            print(f"  recruiter_response_rate: {signals.get('recruiter_response_rate')}")
            
            # Test feature computations live
            depth_feat = SkillDepthFeature()
            val, tag = depth_feat.compute(row)
            print(f"  skill_depth computed live: {val}")
            
            act_feat = ActivityQualityCompositeFeature()
            val, tag = act_feat.compute(row)
            print(f"  activity_composite computed live: {val}")
            
            found += 1
            if found >= 2:
                break

print("\nDone.")
