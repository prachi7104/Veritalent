import os
import json
import time
from explainability_lab.narrative.grounded_narrative_generator import generate_narrative
from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.attribution.feature_contribution_summary import get_top_k_contributions
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

CACHE_DIR = "explainability_lab/narratives_cache"

# Load feature store into memory
feature_store = {}
with open("feature_lab/store/feature_store.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        feature_store[row["candidate_id"]] = row

# Load top-100 from submission
top_100 = []
with open("submission/submission.csv", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(",", 3)
        top_100.append(parts[0])  # candidate_id

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
        candidate_dict = {col: float(features.get(col, 0.0)) for col in TRAINING_FEATURES}
        
        explainer_output = explainer.explain_candidate(candidate_dict)
        shap_summary = get_top_k_contributions(explainer_output, k=5)
        
        retries = 5
        while retries > 0:
            try:
                narrative = generate_narrative(cid, shap_summary, mode="precompute")
                print(f"[{i+1}/{len(missing)}] {cid} — generated ✓")
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                    print(f"[{i+1}/{len(missing)}] {cid} — rate limit hit, retrying in 5s...")
                    time.sleep(5)
                    retries -= 1
                else:
                    print(f"[{i+1}/{len(missing)}] {cid} — failed: {e}, using fallback")
                    break
        
        time.sleep(2.5)  # 2.5s between successful calls to stay under 30 RPM

    print("Done. All candidates now have cached narratives.")
