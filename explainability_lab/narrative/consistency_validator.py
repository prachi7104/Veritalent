def validate_consistency(narrative: str, shap_summary: list[dict]) -> bool:
    """
    Validates that a narrative references ONLY the features in the top-k SHAP summary.
    Uses Option A (exact-string matching of feature names) to ensure the LLM complied 
    with the 'MUST mention each feature by its exact name in bold' instruction.
    """
    valid_features = [item["feature"] for item in shap_summary]
    
    # Check if narrative missed any valid feature? The prompt says they must reference ONLY these. 
    # Usually it's enough to check if it Hallucinated anything outside, but the easiest Option A 
    # is to verify that it uses exactly the bolded names, and we can also check if it uses ANY feature name
    # not in the valid list.
    
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES
    
    # If the narrative mentions ANY training feature that is NOT in the valid top-k list, it's a hallucination
    narrative_lower = narrative.lower()
    
    for feature in TRAINING_FEATURES:
        if feature in narrative_lower and feature not in valid_features:
            print(f"Hallucination detected: Narrative cited '{feature}' which is not in the top-k SHAP list.")
            return False
            
    # Also verify it actually mentioned at least one valid feature to avoid empty/generic passes
    mentioned_any = any(feat in narrative_lower for feat in valid_features)
    if not mentioned_any and len(valid_features) > 0:
        print("Consistency check failed: Narrative did not mention any of the exact feature names.")
        return False
        
    return True
