import os
import sys
import json
import time
from pathlib import Path
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    for _raw in _env_path.read_text(encoding="utf-8").splitlines():
        _raw = _raw.strip()
        if _raw and not _raw.startswith("#") and "=" in _raw:
            _k, _v = _raw.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
JD_CONTEXT = """
Senior AI/ML Engineer Search & Retrieval JD Context:
- Required: 5-9 years experience band.
- Required: "Writes code" (must have active coding signals, e.g., GitHub, not just management).
- Preferred: Product-company background (services-firm only is a risk flag, but acceptable if they have prior product history).
- Preferred: Location Pune/Noida.
- Preferred: Sub-30-day notice period.
- Required Core Skills: Deep IR depth (Ranking Systems, Text Encoders, Vector Representations, Search Backend).
"""

PROMPT_TEMPLATE = """
You are an expert technical recruiter evaluating a candidate for a Senior AI/ML Engineer (Search & Retrieval) role.

{jd_context}

CANDIDATE PROFILE:
{candidate_json}

Evaluate the candidate and assign a relevance label from 0 to 3 based on these definitions:
0 = Irrelevant (HR manager, accountant, unrelated title)
1 = Plausible but weak (ML-adjacent but missing core IR skills)
2 = Strong fit (ML engineer with some deep-IR skills, product experience)
3 = Excellent fit (Senior/Staff AI with deep-IR depth, active, product background)

Respond ONLY with valid JSON in this exact format, with no preamble or markdown blocks outside the JSON:
{{"label": <0, 1, 2, or 3>, "reason": "<one sentence explaining the score based on the candidate's summary, title, career history, and IR skills>"}}
"""

def generate_llm_labels(sample_path: str, output_path: str):
    import openai
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    cerebras_api_key = os.environ.get("CEREBRAS_API_KEY")
    
    if not groq_api_key and not cerebras_api_key:
        print("ERROR: GROQ_API_KEY or CEREBRAS_API_KEY required.", file=sys.stderr)
        sys.exit(1)
        
    client_groq = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        timeout=30.0  # 30 second timeout as requested
    ) if groq_api_key else None
    
    client_cerebras = openai.OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=cerebras_api_key,
        timeout=30.0
    ) if cerebras_api_key else None

    labels = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            
    with open(sample_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
        
    missing_sample = [c for c in sample if c["candidate_id"] not in labels]
    print(f"Loaded {len(labels)} existing labels. Processing {len(missing_sample)} remaining candidates...")
    
    active_api = "groq" if client_groq else "cerebras"
    
    for i, cand in enumerate(missing_sample):
        print(f"Scoring {i+1}/{len(missing_sample)} via {active_api}...")
        prompt = PROMPT_TEMPLATE.format(
            jd_context=JD_CONTEXT,
            candidate_json=json.dumps(cand, indent=2)
        )
        
        max_retries = 3
        base_wait = 5
        success = False
        
        for attempt in range(max_retries):
            try:
                response_text = ""
                if active_api == "groq":
                    chat_completion = client_groq.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.0
                    )
                    response_text = chat_completion.choices[0].message.content
                elif active_api == "cerebras":
                    chat_completion = client_cerebras.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="gpt-oss-120b",
                        temperature=0.0
                    )
                    response_text = chat_completion.choices[0].message.content
                    
                clean_json = response_text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                labels[cand["candidate_id"]] = {
                    "label": parsed["label"],
                    "reason": parsed["reason"]
                }
                
                print(f"  -> Label: {parsed['label']} | Reason: {parsed['reason']}")
                sys.stdout.flush()
                
                # Small sleep to be polite
                time.sleep(1) 
                success = True
                break # Success
                
            except Exception as e:
                err_str = str(e).lower()
                is_timeout = "timeout" in err_str
                is_rate_limit = "429" in err_str or "503" in err_str or "quota" in err_str
                
                if is_timeout or is_rate_limit:
                    if active_api == "groq" and client_cerebras:
                        print(f"\n[!] Groq hit timeout/rate-limit. Switching to Cerebras gpt-oss-120b instantly!")
                        active_api = "cerebras"
                        # Do not break, loop will immediately retry with Cerebras
                        continue
                    else:
                        wait_time = base_wait * (2 ** attempt)
                        print(f"Rate limit or timeout on {active_api}. Sleeping {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                else:
                    print(f"Error scoring candidate {cand['candidate_id']}: {e}", file=sys.stderr)
                    labels[cand["candidate_id"]] = {"label": 1, "reason": f"Error: {e}"}
                    success = True
                    break
        
        if not success:
            print(f"Failed to score candidate {cand['candidate_id']} after {max_retries} retries.", file=sys.stderr)
            labels[cand["candidate_id"]] = {"label": 1, "reason": "Error: Max retries exhausted due to timeouts."}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
        
    print(f"\nCompleted! Generated {len(labels)} labels.")

if __name__ == "__main__":
    generate_llm_labels(
        r"c:\projects\Veritalent\ranking_lab\labels\stratified_sample.json",
        r"c:\projects\Veritalent\ranking_lab\labels\llm_labels.json"
    )
