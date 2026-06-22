FEATURE_SCHEMA = {
    "candidate_id": str,
    
    "skill_depth": float,
    "skill_depth_reliability": str,
    "skill_breadth": float,
    "skill_breadth_reliability": str,
    "skill_recency": float,
    "skill_recency_reliability": str,
    "skill_mastery_triangulation": float,
    "skill_mastery_triangulation_reliability": str,
    
    "career_velocity": float,
    "career_velocity_reliability": str,
    "promotion_velocity": float,
    "promotion_velocity_reliability": str,
    "tenure_stability": float,
    "tenure_stability_reliability": str,
    "inflection_point_strength": float,
    "inflection_point_strength_reliability": str,
    
    "trust_score": float,
    "trust_reasons": list,
    "trust_score_composite_reliability": str,
    
    "activity_quality_composite": float,
    "activity_quality_composite_reliability": str,
    
    "industry_relevance": float,
    "industry_relevance_reliability": str,
    
    "logistics_fit_score": float,
    "logistics_fit_score_reliability": str,
    
    "product_vs_services": float,
    "product_vs_services_reliability": str
}

def validate_schema(record: dict) -> bool:
    """Verify that a record conforms to the defined schema."""
    for key, expected_type in FEATURE_SCHEMA.items():
        if key not in record:
            return False
        # allow None for list/dict types if they might be empty in dict form,
        # but typically we'll just check type if it's not None.
        if record[key] is not None and not isinstance(record[key], expected_type):
            # Special case for floats matching ints
            if expected_type == float and isinstance(record[key], int):
                continue
            return False
    return True
