import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ranking_lab.experiments.common import load_feature_store
from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.attribution.feature_contribution_summary import get_top_k_contributions
from explainability_lab.narrative.grounded_narrative_generator import generate_narrative, generate_ungrounded_narrative
from explainability_lab.narrative.consistency_validator import validate_consistency
from explainability_lab.narrative.fallback_narrative import generate_fallback

def run_faithfulness_eval(n_samples: int = 20):
    """
    Compares the faithfulness of SHAP-grounded narratives vs ungrounded LLM narratives.
    """
    feature_store = load_feature_store()
    candidate_ids = list(feature_store.keys())[:n_samples]
    
    explainer = SHAPExplainer()
    
    # We also need raw candidate data for the ungrounded model to look at profile/skills
    raw_cands = []
    try:
        with open("dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if c["candidate_id"] in candidate_ids:
                    raw_cands.append(c)
                if len(raw_cands) >= n_samples:
                    break
    except FileNotFoundError:
        print("Warning: could not load raw candidates, ungrounded will use empty profile.")
        raw_cands = [{"candidate_id": cid, "profile": {}, "skills": []} for cid in candidate_ids]
        
    cand_dict = {c["candidate_id"]: c for c in raw_cands}
    
    results = {
        "grounded_hallucination_count": 0,
        "ungrounded_hallucination_count": 0,
        "total": len(candidate_ids),
        "examples": []
    }
    
    for cid in candidate_ids:
        print(f"Evaluating {cid}...")
        features = feature_store[cid]
        
        shap_out = explainer.explain_candidate(features)
        top_k = get_top_k_contributions(shap_out, k=5)
        
        # Grounded
        try:
            grounded_text = generate_narrative(cid, top_k, mode="precompute")
        except Exception as e:
            print(f"LLM Error: {e}")
            continue
            
        grounded_valid = validate_consistency(grounded_text, top_k)
        if not grounded_valid:
            results["grounded_hallucination_count"] += 1
            
        # Ungrounded
        raw_data = cand_dict.get(cid, {})
        try:
            ungrounded_text = generate_ungrounded_narrative(cid, raw_data)
        except Exception as e:
            print(f"LLM Error: {e}")
            continue
            
        ungrounded_valid = validate_consistency(ungrounded_text, top_k)
        if not ungrounded_valid:
            results["ungrounded_hallucination_count"] += 1
            
        if len(results["examples"]) < 3:
            results["examples"].append({
                "candidate_id": cid,
                "top_k_features": [f["feature"] for f in top_k],
                "grounded": grounded_text,
                "grounded_valid": grounded_valid,
                "ungrounded": ungrounded_text,
                "ungrounded_valid": ungrounded_valid
            })
            
        time.sleep(1) # Rate limit
        
    grounded_rate = results["grounded_hallucination_count"] / results["total"]
    ungrounded_rate = results["ungrounded_hallucination_count"] / results["total"]
    
    print("\n--- Faithfulness Results ---")
    print(f"Grounded Hallucination Rate: {grounded_rate*100:.1f}%")
    print(f"Ungrounded Hallucination Rate: {ungrounded_rate*100:.1f}%")
    
    from explainability_lab.comparison.readability_eval import run_readability_eval
    readability = run_readability_eval(results)
    
    report_path = os.path.join(os.path.dirname(__file__), '../reports/explainability_comparison_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report_content = f"""# Explainability Comparison Report

## Faithfulness Results
- Evaluated {n_samples} candidates.
- **Grounded Hallucination Rate:** {grounded_rate*100:.1f}%
- **Ungrounded Hallucination Rate:** {ungrounded_rate*100:.1f}%

## Readability Results
- **Template-based Fallback:** {readability['avg_template_sentence_length']:.1f} avg words/sentence
- **LLM Grounded:** {readability['avg_llm_sentence_length']:.1f} avg words/sentence

## Recommendations
- **Live Demo Path:** Precomputed `serve` mode (hits cache or uses `fallback_narrative`).
- **Demo Prep:** `precompute` mode to populate the cache with LLM-grounded narratives.
- **Future Production:** Use `grounded_narrative_generator` with strict Option A consistency validation. Any failed validations must trigger the `fallback_narrative`.

## Important Limitation: Explanation Faithfulness vs. Candidate Quality

These explanations faithfully reflect what the trained model learned from
LLM-judged training labels. They do NOT constitute independent ground-truth
assessments of candidate quality. A high-ranked candidate's explanation
accurately describes why the MODEL scored them highly — not why a human
recruiter would necessarily agree. The model's known limitation (keyword
stuffer adversarial test FAILED in Lab 06) applies equally here: a
narrative may read as compelling for a candidate the model incorrectly
ranked highly.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nReport written to {report_path}")
    return results

if __name__ == "__main__":
    run_faithfulness_eval(5)
