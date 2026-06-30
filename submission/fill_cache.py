import os
import sys
import json
import time
import csv
from pathlib import Path

# Fix relative imports when running from the script directory
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from explainability_lab.narrative.grounded_narrative_generator import generate_narrative
from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.attribution.feature_contribution_summary import get_top_k_contributions
from explainability_lab.narrative.candidate_context import build_candidate_context
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

CACHE_DIR = "explainability_lab/narratives_cache"

# Load feature store into memory
feature_store = {}
pool_jd_skill_sum = 0.0
with open("feature_lab/store/feature_store.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        feature_store[row["candidate_id"]] = row
        pool_jd_skill_sum += float(row.get('jd_skill_score', 0) or 0)
pool_jd_skill_mean = pool_jd_skill_sum / max(1, len(feature_store))

# Load candidates
candidates = {}
candidates_file = 'dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl'
with open(candidates_file, 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        candidates[row['candidate_id']] = row

# Load top-100 from submission
top_100 = []
ranks = {}
with open("submission/submission.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = row["candidate_id"]
        top_100.append(cid)
        ranks[cid] = int(row["rank"])

# Find which ones don't have cached narratives yet
missing = [
    cid for cid in top_100
    if not os.path.exists(f"{CACHE_DIR}/{cid}.json")
]

print(f"Candidates needing LLM narratives: {len(missing)}")

if not missing:
    print("All candidates have cached narratives.")
else:
    # Load SHAP explainer once
    explainer = SHAPExplainer("ranking_lab/models/gbm_lambdarank.txt")

    # Regenerate with rate limit handling
    for i, cid in enumerate(missing):
        features = feature_store.get(cid, {})
        candidate_data = candidates.get(cid, {})
        candidate_dict = {col: float(features.get(col, 0.0)) for col in TRAINING_FEATURES}
        
        explainer_output = explainer.explain_candidate(candidate_dict)
        
        # Format contributions list
        shap_summary_list = []
        for f_name, details in explainer_output["contributions"].items():
            shap_summary_list.append({
                "feature": f_name,
                "raw_value": details["raw_value"],
                "shap_value": details["shap_value"]
            })
            
        shap_summary = get_top_k_contributions(explainer_output, k=5)
        
        context = build_candidate_context(
            candidate=candidate_data,
            features=features,
            shap_contributions=shap_summary,
            rank=ranks.get(cid, 100),
            pool_jd_skill_mean=pool_jd_skill_mean
        )
        
        retries = 5
        while retries > 0:
            try:
                narrative = generate_narrative(cid, context, mode="precompute")
                print(f"[{i+1}/{len(missing)}] {cid} - generated")
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                    print(f"[{i+1}/{len(missing)}] {cid} - rate limit hit, retrying in 5s...")
                    time.sleep(5)
                    retries -= 1
                else:
                    print(f"[{i+1}/{len(missing)}] {cid} - failed: {e}, using fallback")
                    break
        
        time.sleep(2.5)  # 2.5s between successful calls to stay under 30 RPM

    print("Done. All candidates now have cached narratives.")
