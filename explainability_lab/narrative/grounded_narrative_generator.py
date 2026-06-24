import os
import json
import openai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup client as requested
pythonclient = openai.OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

CACHE_DIR = Path("explainability_lab/narratives_cache")

def generate_narrative(candidate_id: str, shap_summary: list[dict], mode: str = "precompute", model: str = "gpt-oss-120b") -> str:
    """
    Generates a SHAP-grounded narrative using two modes:
      - mode='precompute': Calls the LLM and caches the result.
      - mode='serve': Checks cache; if hit returns it, if miss returns fallback immediately (no LLM call).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{candidate_id}.json"
    
    if mode == "serve":
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)["narrative"]
        else:
            from explainability_lab.narrative.fallback_narrative import generate_fallback
            return generate_fallback(shap_summary)
            
    # mode == "precompute"
    features_str = "\n".join([f"- Feature: **{item['feature']}** (Value: {item['raw_value']}, SHAP contribution: {item['shap_value']:.4f})" for item in shap_summary])
    
    prompt = f"""You are a recruiter-facing AI assistant. Explain why this candidate received their ranking score.
    
Here are the TOP contributing factors from the model's decision process (SHAP values):
{features_str}

RULES:
1. You MUST reference ONLY these provided features. Do NOT hallucinate or add outside reasoning.
2. You MUST mention each feature by its exact name in bold, e.g. **skill_depth**, **trust_score**.
3. Keep it brief (2-4 sentences).

Draft the explanation now:
"""

    response = pythonclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    narrative = response.choices[0].message.content.strip()
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"candidate_id": candidate_id, "narrative": narrative}, f)
        
    return narrative

def generate_ungrounded_narrative(candidate_id: str, candidate_data: dict, model: str = "gpt-oss-120b") -> str:
    """
    Generates an ungrounded baseline narrative for comparison, given raw candidate data but NO SHAP values.
    """
    prompt = f"""You are a recruiter-facing AI assistant. Explain why this candidate is a good or bad fit.
    
Candidate Profile:
{json.dumps(candidate_data.get('profile', {}))}
Skills: {json.dumps(candidate_data.get('skills', []))}

Draft a 2-4 sentence explanation of why they rank where they do.
"""
    response = pythonclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
