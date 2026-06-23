def check_assessment_corroboration(candidate: dict) -> dict:
    """
    Computes delta between self-reported proficiency and objective assessment scores.
    Candidates with no assessment data will have an output indicating zero weight.
    """
    redrob_signals = candidate.get("redrob_signals", {})
    assessment_scores = redrob_signals.get("skill_assessment_scores", {})
    
    if not assessment_scores:
        return {
            "has_assessments": False,
            "average_delta": 0.0,
            "weight_multiplier": 0.0, # Zero weight explicitly
            "details": "No assessment data available"
        }
        
    # Map proficiency to expected score ranges (approximate for delta calculation)
    # Master context: "advanced" mean 52.8, "expert" mean 71.4
    prof_map = {
        "beginner": 20.0,
        "intermediate": 40.0,
        "advanced": 60.0,
        "expert": 80.0
    }
    
    skills = {s.get("name"): s.get("proficiency", "").lower() for s in candidate.get("skills", [])}
    
    deltas = []
    details = []
    for skill_name, score in assessment_scores.items():
        if skill_name in skills:
            prof = skills[skill_name]
            expected = prof_map.get(prof, None)
            if expected is not None:
                # Delta measures over-claiming: expected minus scored
                delta = expected - score
                deltas.append(delta)
                details.append(f"{skill_name}: claimed {prof}, scored {score} (delta {delta})")
                
    if not deltas:
        return {
            "has_assessments": False,
            "average_delta": 0.0,
            "weight_multiplier": 0.0,
            "details": "Assessment data present but no matching skills found"
        }
        
    avg_delta = sum(deltas) / len(deltas)
    
    return {
        "has_assessments": True,
        "average_delta": float(avg_delta),
        "weight_multiplier": 1.0,
        "details": "; ".join(details)
    }
