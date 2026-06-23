def check_yoe_consistency(candidate: dict) -> dict:
    """
    Computes the deviation between stated years_of_experience and the 
    sum of duration_months across all career_history entries.
    
    Returns:
        dict: {
            "deviation_years": float,
            "flagged": bool,
            "details": str
        }
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience")
    if yoe is None:
        yoe = 0.0
    else:
        try:
            yoe = float(yoe)
        except ValueError:
            yoe = 0.0
        
    career_history = candidate.get("career_history", [])
    total_months = sum(entry.get("duration_months", 0) for entry in career_history)
    calculated_yoe = total_months / 12.0
    
    deviation = abs(yoe - calculated_yoe)
    flagged = deviation > 1.5
    
    return {
        "deviation_years": float(deviation),
        "flagged": flagged,
        "details": f"Stated YOE: {yoe:.2f}, Calculated from history: {calculated_yoe:.2f}, Deviation: {deviation:.2f} years"
    }
