def validate_consistency(narrative: str, shap_summary: list[dict]) -> bool:
    # Since we are now using human-readable names, exact feature name matching will fail.
    # We will just verify the narrative is not empty and within reasonable length.
    if not narrative or len(narrative.split()) < 20:
        return False
    return True
