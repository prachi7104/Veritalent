import os
import json
import openai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup client as requested
if os.environ.get("GROQ_API_KEY"):
    pythonclient = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
        max_retries=2
    )
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
else:
    pythonclient = openai.OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=os.environ.get("CEREBRAS_API_KEY"),
        max_retries=0
    )
    DEFAULT_MODEL = "gpt-oss-120b"

CACHE_DIR = Path("explainability_lab/narratives_cache")

SYSTEM_PROMPT = """You are a senior technical recruiter writing candidate evaluation notes.
Your job is to explain why a candidate ranked where they did for a specific role.

Rules:
1. Write in plain English. No raw feature names like "skill_mastery_triangulation" —
   translate them: skill_mastery_triangulation → "verified skill depth",
   skill_depth → "breadth of technical skills", logistics_fit_score → "availability and location fit",
   activity_quality_composite → "platform engagement", implied_skill_score → "IR vocabulary in profile",
   yoe_band_fit → "experience level alignment", jd_skill_score → "JD-relevant skill match",
   trust_score → "profile credibility".
2. Lead with WHO the candidate is: their current role, YOE, and company background.
3. State the primary ranking driver in one sentence.
4. Name 1–2 JD-specific alignment points (the JD is for a Senior AI/ML Engineer
   focused on Search & Retrieval, 5–9 YOE, Pune/Noida, product company preferred).
5. Add one honest limitation or caveat at the end.
6. Maximum 120 words. Be precise. Be honest. Do not hallucinate facts not in the data.
7. Do not start with "The candidate" — vary your openings.
"""

USER_PROMPT_TEMPLATE = """
Role: Senior AI/ML Engineer — Search & Retrieval (5–9 YOE, Pune/Noida, product-first)
Rank: #{rank} of 100

Candidate snapshot:
- Current title: {current_title}
- Years of experience: {yoe}
- Current company: {current_company} ({company_type})
- Location: {location}
- Notice period: {notice_period}
- Key skills (JD-relevant): {jd_skills}
- Top SHAP drivers: {shap_summary}
- JD skill score: {jd_skill_score:.1f} (pool mean: {pool_jd_skill_mean:.1f})
- Experience band: {yoe_band_label}
- Trust status: {trust_label}

Write a 80–120 word recruiter note explaining this ranking.
Start with who this person is. Be specific. Be honest. No bullet points.
"""

def generate_narrative(candidate_id: str, context: dict, mode: str = "precompute", model: str = None) -> str:
    if model is None:
        model = DEFAULT_MODEL
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
            from explainability_lab.narrative.fallback_narrative import generate_fallback_narrative
            return generate_fallback_narrative(context)
            
    # mode == "precompute"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["narrative"]
            
    prompt = USER_PROMPT_TEMPLATE.format(**context)
    
    response = pythonclient.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        timeout=5.0
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
