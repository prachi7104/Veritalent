# submission/batch_scoring_pipeline.py
"""
Batch Scoring Pipeline v2 — updated to use BlendScorer and pre-cached narratives.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Fix relative imports when running from the script directory
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from submission.blend_scorer import BlendScorer
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES as FEATURE_COLS
from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.attribution.feature_contribution_summary import get_top_k_contributions
from explainability_lab.narrative.grounded_narrative_generator import generate_narrative
from explainability_lab.narrative.fallback_narrative import generate_fallback_narrative
from explainability_lab.narrative.consistency_validator import validate_consistency
from explainability_lab.narrative.candidate_context import build_candidate_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission/submission.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top candidates to export")
    args = parser.parse_args()

    print("Loading feature store...")
    feature_store = {}
    pool_jd_skill_sum = 0.0
    
    # Load feature store dynamically, selecting the larger file
    path_v2 = Path("feature_lab/store/feature_store_v2.jsonl")
    path_full = Path("feature_lab/store/feature_store.jsonl")
    store_path = path_full if (path_full.exists() and (not path_v2.exists() or path_full.stat().st_size > path_v2.stat().st_size)) else path_v2
        
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            feature_store[row['candidate_id']] = row
            pool_jd_skill_sum += float(row.get('jd_skill_score', 0) or 0)
            
    print(f"Loaded {len(feature_store)} candidates from {store_path}.")
    pool_jd_skill_mean = pool_jd_skill_sum / max(1, len(feature_store))
    
    print("Loading candidates...")
    candidates = {}
    candidates_file = 'dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl'
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            candidates[row['candidate_id']] = row

    print("Initializing BlendScorer...")
    scorer = BlendScorer()

    print("Scoring all candidates...")
    candidate_ids = list(feature_store.keys())
    scores_dict = scorer.score(candidate_ids, feature_store)

    scores = [(cid, score) for cid, score in scores_dict.items()]
    print(f"Successfully scored {len(scores)} candidates.")
    
    print("Sorting candidates by blend score...")
    # Sort by score descending, then candidate_id ascending for tie-breaks
    scores.sort(key=lambda x: (-x[1], x[0]))
    
    top_n_candidates = scores[:args.top_n]

    # Load pre-cached narratives if available
    narratives_cache = {}
    cache_path = Path("submission/narratives_cache.json")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            narratives_cache = json.load(f)
        print(f"Loaded {len(narratives_cache)} narratives from cache.")

    print("Initializing Explainer...")
    explainer = SHAPExplainer(model_path="ranking_lab/models/gbm_lambdarank.txt")
    
    print("Preparing submission rows...")
    rows = []
    
    for rank, (candidate_id, score) in enumerate(top_n_candidates, start=1):
        features = feature_store[candidate_id]
        
        # Check if rank is cached
        reasoning = None
        if narratives_cache:
            # Try rank string first
            reasoning_entry = narratives_cache.get(str(rank))
            if reasoning_entry:
                reasoning = reasoning_entry.get("narrative")
            else:
                # Try candidate_id key
                reasoning_entry = narratives_cache.get(candidate_id)
                if reasoning_entry:
                    reasoning = reasoning_entry.get("narrative")

        if reasoning:
            rows.append({
                "candidate_id": candidate_id,
                "rank": rank,
                "score": round(score, 6),
                "reasoning": reasoning
            })
            continue

        # Fallback: compute and generate narrative if not in cache
        print(f"Cache miss for rank {rank} ({candidate_id}). Generating...")
        candidate_dict = {}
        for col in FEATURE_COLS:
            candidate_dict[col] = float(features.get(col, 0.0) or 0.0)
            
        explainer_output = explainer.explain_candidate(candidate_dict)
        
        # Format contributions list
        shap_summary = []
        for f_name, details in explainer_output["contributions"].items():
            shap_summary.append({
                "feature": f_name,
                "raw_value": details["raw_value"],
                "shap_value": details["shap_value"]
            })
        
        context = build_candidate_context(
            candidate=candidates.get(candidate_id, {}),
            features=features,
            shap_contributions=shap_summary,
            rank=rank,
            pool_jd_skill_mean=pool_jd_skill_mean
        )
        
        try:
            reasoning = generate_narrative(
                candidate_id=candidate_id,
                context=context,
                mode="precompute"
            )
            if not validate_consistency(reasoning, shap_summary):
                reasoning = generate_fallback_narrative(context)
        except Exception as e:
            print(f"Exception for rank {rank}: {e}")
            reasoning = generate_fallback_narrative(context)

        rows.append({
            "candidate_id": candidate_id,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning
        })

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing {args.output}...")
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
        
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
