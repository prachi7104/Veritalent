import re

def check_template_reliance(candidate: dict, known_templates: set = None) -> dict:
    """
    EXPERIMENTAL SIGNAL. 
    This signal detects description-template-reliance. It is a soft signal only and risks 
    mislabeling concise, legitimate candidates as low-trust.
    """
    if known_templates is None:
        known_templates = set()
        
    texts = []
    profile = candidate.get("profile", {})
    if profile.get("summary"):
        texts.append(profile.get("summary"))
        
    for entry in candidate.get("career_history", []):
        if entry.get("description"):
            texts.append(entry.get("description"))
            
    if not texts:
        return {"template_fraction": 0.0, "details": "No free text available"}
        
    def normalize(t):
        return re.sub(r'\s+', ' ', t.lower().strip())
        
    match_count = 0
    for t in texts:
        norm_t = normalize(t)
        # Near-exact match: if a known template is a substring or vice-versa, or exact.
        for kt in known_templates:
            if not kt:
                continue
            norm_kt = normalize(kt)
            # Threshold could be length based to avoid matching very short strings
            if len(norm_kt) > 20 and (norm_kt in norm_t or norm_t in norm_kt):
                match_count += 1
                break
                
    fraction = match_count / len(texts)
    return {
        "template_fraction": float(fraction),
        "details": f"{match_count}/{len(texts)} texts match templates"
    }
