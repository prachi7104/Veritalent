def generate_trust_explanation(trust_details: dict) -> str:
    """
    Converts trust score breakdown details into human-readable flags.
    MUST include the caveat: "Does NOT catch sophisticated consistent fraud."
    """
    flags = []
    
    # YOE Risk
    yoe = trust_details.get("yoe", {})
    if yoe.get("flagged"):
        flags.append(f"Stated {yoe.get('stated_yoe', 0):.1f} years of experience does not match listed work history ({yoe.get('calculated_yoe', 0):.1f} years) — a deviation of {yoe.get('deviation_years', 0):.1f} years.")
        
    # Proficiency Risk
    prof = trust_details.get("prof", {})
    if prof.get("flagged"):
        flags.append(f"Claimed 'expert' level on skills with low evidence (Severity sum: {prof.get('severity_sum', 0)}).")
        
    # Template Risk
    temp = trust_details.get("temp", {})
    if temp.get("flagged"):
        flags.append(f"High template reliance: {temp.get('template_fraction', 0.0)*100:.1f}% of descriptions match known boilerplate.")
        
    # Identity Risk
    ident = trust_details.get("ident", {})
    if ident.get("flagged"):
        flags.append(f"Identity unverified: {ident.get('unverified_fraction', 0.0)*100:.1f}% of required checks failed.")
        
    base = "Trust Score Analysis:\n"
    if not flags:
        base += "No significant anomalies detected in candidate profile."
    else:
        base += "\n".join([f"- {f}" for f in flags])
        
    base += "\n\n*Note: This system does NOT catch sophisticated consistent fraud.*"
    return base
