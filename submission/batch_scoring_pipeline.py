import json
import csv
import lightgbm as lgb
import os
from pathlib import Path

# Fix relative imports when running from the script directory
import sys
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES as FEATURE_COLS
from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.attribution.feature_contribution_summary import get_top_k_contributions
from explainability_lab.narrative.grounded_narrative_generator import generate_narrative
from explainability_lab.narrative.fallback_narrative import generate_fallback
from explainability_lab.narrative.consistency_validator import validate_consistency

def main():
    print("Loading feature store...")
    feature_store = {}
    with open('feature_lab/store/feature_store.jsonl', 'r') as f:
        for line in f:
            row = json.loads(line)
            feature_store[row['candidate_id']] = row
            
    print(f"Loaded {len(feature_store)} candidates from feature store.")
    
    print("Loading GBM model...")
    model_path = "ranking_lab/models/gbm_lambdarank.txt"
    model = lgb.Booster(model_file=model_path)
    
    print("Scoring candidates...")
    scores = []
    missing_from_feature_store = []
    
    for candidate_id, features in feature_store.items():
        try:
            feature_vector = [float(features.get(col, 0.0)) for col in FEATURE_COLS]
            score = model.predict([feature_vector])[0]
            scores.append((candidate_id, score))
        except Exception as e:
            missing_from_feature_store.append(candidate_id)
            
    print(f"Successfully scored {len(scores)} candidates.")
    print(f"Failed to score {len(missing_from_feature_store)} candidates.")
    
    if len(missing_from_feature_store) > 100:
        print("ERROR: Too many candidates failed scoring. Aborting.")
        sys.exit(1)
        
    print("Sorting candidates...")
    # Sort by score descending, then candidate_id ascending for tie-breaks
    scores.sort(key=lambda x: (-x[1], x[0]))
    
    top_100 = scores[:100]
    
    print("Initializing Explainer...")
    explainer = SHAPExplainer(model_path=model_path)
    
    print("Generating narratives and building submission rows...")
    rows = []
    os.makedirs("submission", exist_ok=True)
    
    # Pre-calculate SHAP summaries sequentially (since tree explainer might not be thread safe)
    candidate_data = []
    for rank, (candidate_id, score) in enumerate(top_100, start=1):
        features = feature_store[candidate_id]
        
        candidate_dict = {}
        for col in FEATURE_COLS:
            candidate_dict[col] = float(features.get(col, 0.0))
            
        explainer_output = explainer.explain_candidate(candidate_dict)
        shap_summary = get_top_k_contributions(explainer_output, k=5)
        
        candidate_data.append({
            "candidate_id": candidate_id,
            "rank": rank,
            "score": score,
            "shap_summary": shap_summary
        })
        
    # Generate narratives concurrently
    import concurrent.futures

    def process_candidate(data):
        rank = data["rank"]
        candidate_id = data["candidate_id"]
        shap_summary = data["shap_summary"]
        score = data["score"]
        
        try:
            reasoning = generate_narrative(
                candidate_id=candidate_id,
                shap_summary=shap_summary,
                mode="precompute",
                model="gpt-oss-120b"
            )
            if not validate_consistency(reasoning, shap_summary):
                reasoning = generate_fallback(shap_summary)
        except Exception as e:
            print(f"Exception for rank {rank}: {e}")
            reasoning = generate_fallback(shap_summary)
            
        return {
            "candidate_id": candidate_id,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning
        }

    print("Submitting to ThreadPoolExecutor...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_data = {executor.submit(process_candidate, d): d for d in candidate_data}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_data), start=1):
            try:
                res = future.result()
            except Exception as e:
                print(f"Future raised exception: {e}")
                continue
            rows.append(res)
            if i % 10 == 0:
                print(f"Generated {i}/100 narratives...")
                
    # Re-sort rows by rank since they complete out of order
    rows.sort(key=lambda x: x["rank"])
            
    print("Writing submission.csv...")
    with open('submission/submission.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
        
    print("Pipeline complete!")

if __name__ == "__main__":
    main()
