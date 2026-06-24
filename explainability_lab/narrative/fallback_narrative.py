def generate_fallback(shap_summary: list[dict]) -> str:
    """
    Purely template-based generator. Zero external dependencies.
    Used when caching fails or consistency validator flags a hallucination.
    """
    if not shap_summary:
        return "Ranked based on model baseline."
        
    parts = []
    for item in shap_summary:
        # e.g., skill_depth (4.5)
        sign = "+" if item["shap_value"] > 0 else "-"
        parts.append(f"{item['feature']} ({item['raw_value']}, impact: {sign})")
        
    return f"Ranked primarily due to: {', '.join(parts)}."
