import os
import json
import hashlib
import logging
import httpx
import re

logger = logging.getLogger(__name__)

CACHE_DIR = "backend/cache/jd_decompositions"

def get_decomposition(jd_text: str) -> dict:
    return _get_decomp(jd_text, allow_llm=True)

def get_decomposition_no_llm(jd_text: str) -> dict:
    return _get_decomp(jd_text, allow_llm=False)

def _get_decomp(jd_text: str, allow_llm: bool) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.sha256(jd_text.encode('utf-8')).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{h}.json")

    if os.path.exists(cache_path):
        logger.info(f"jd_decomp cache hit for {h}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"jd_decomp cache miss for {h}")

    if allow_llm:
        res = _try_llm_decomposition(jd_text)
        if res:
            res["fallback_used"] = False
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(res, f)
            return res

    # Keyword extraction fallback
    return _keyword_fallback(jd_text)

def _try_llm_decomposition(jd_text: str) -> dict | None:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        logger.warning("No LLM API key available, using fallback")
        return None

    base_url = "https://api.groq.com/openai/v1/chat/completions" if os.environ.get("GROQ_API_KEY") else "https://api.cerebras.ai/v1/chat/completions"
    model = "llama-3.3-70b-versatile" if os.environ.get("GROQ_API_KEY") else "gpt-oss-120b"
    
    prompt = f"""
You are an expert recruiter. Decompose this Job Description into JSON format.
JD: {jd_text}

Extract the following keys exactly:
- must_haves: list of strings (required skills/tech)
- nice_to_haves: list of strings (preferred skills)
- hard_exclusions: list of strings (dealbreakers)
- experience_band: string (e.g. "5-9 years")
- logistics: dict of logistics requirements (e.g. location, notice period)

Respond ONLY with valid JSON.
"""

    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.post(
                base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            
            clean_json = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            
            return {
                "must_haves": parsed.get("must_haves", []),
                "nice_to_haves": parsed.get("nice_to_haves", []),
                "hard_exclusions": parsed.get("hard_exclusions", []),
                "experience_band": parsed.get("experience_band", "Unknown"),
                "logistics": parsed.get("logistics", {})
            }
    except Exception as e:
        logger.warning(f"LLM decomposition failed: {e}")
        return None

def _keyword_fallback(jd_text: str) -> dict:
    words = re.findall(r'\b[A-Z][a-zA-Z0-9+#.]+\b', jd_text)
    unique_words = list(dict.fromkeys(words))
    return {
        "must_haves": unique_words,
        "nice_to_haves": [],
        "hard_exclusions": [],
        "experience_band": "Unknown",
        "logistics": {},
        "fallback_used": True
    }
