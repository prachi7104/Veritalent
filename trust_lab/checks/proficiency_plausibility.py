def check_proficiency_plausibility(candidate: dict) -> dict:
    """
    Checks for implausible skill claims (e.g., expert with 0 duration and 0 endorsements).
    Returns count and severity-weighted sum.
    """
    skills = candidate.get("skills", [])
    implausible_count = 0
    severity_sum = 0.0
    details = []
    
    for skill in skills:
        prof = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)
        
        # Rule: proficiency = expert/advanced/intermediate but with 0 duration and 0 endorsements.
        # "expert with 0 duration and 0 endorsements" is specifically mentioned.
        if duration == 0 and endorsements == 0:
            if prof == "expert":
                implausible_count += 1
                severity_sum += 1.0
                details.append(f"{skill.get('name')} (Expert, 0 duration, 0 endorsements)")
            elif prof == "advanced":
                implausible_count += 1
                severity_sum += 0.5
                details.append(f"{skill.get('name')} (Advanced, 0 duration, 0 endorsements)")
            elif prof == "intermediate":
                implausible_count += 1
                severity_sum += 0.2
                details.append(f"{skill.get('name')} (Intermediate, 0 duration, 0 endorsements)")
                
    # Specific honeypot rule: 3+ skills rated "expert" with 0 months duration.
    honeypot_flag = sum(1 for s in skills if s.get("proficiency", "").lower() == "expert" and s.get("duration_months", 0) == 0) >= 3
    
    return {
        "implausible_count": implausible_count,
        "severity_sum": float(severity_sum),
        "flagged_honeypot_pattern": honeypot_flag,
        "details": "; ".join(details) if details else "No implausible claims"
    }
