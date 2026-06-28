"""
Translates raw SHAP feature contributions into recruiter-readable summaries.
Called before passing SHAP data to the narrative LLM.
"""

FEATURE_LABELS = {
    "skill_mastery_triangulation": "verified skill depth",
    "skill_depth":                 "breadth of technical skills",
    "skill_breadth":               "skill diversity across IR domains",
    "skill_recency":               "recency of IR skill usage",
    "jd_skill_score":              "JD-aligned skill strength",
    "yoe_band_fit":                "experience level alignment",
    "tenure_stability":            "career stability",
    "activity_quality_composite":  "platform engagement signals",
    "trust_score":                 "profile credibility",
    "logistics_fit_score":         "location and availability fit",
    "product_vs_services":         "product-company background",
    "implied_skill_score":         "IR terminology in profile",
}


def format_shap_for_narrative(shap_contributions: list[dict]) -> str:
    """
    Convert [{feature, shap_value}, ...] to a readable string.
    Example output:
      "JD-aligned skill strength (+4.2, primary driver),
       experience level alignment (+1.1),
       profile credibility concern (−0.3)"
    """
    parts = []
    for i, item in enumerate(shap_contributions[:5]):
        feat = item.get("feature", "unknown")
        val = float(item.get("shap_value", 0))
        label = FEATURE_LABELS.get(feat, feat)
        sign = "+" if val >= 0 else "−"
        qualifier = " (primary driver)" if i == 0 else ""
        qualifier += " (concern)" if val < 0 and "trust" in feat else ""
        parts.append(f"{label} ({sign}{abs(val):.2f}{qualifier})")
    return ", ".join(parts)


def get_yoe_band_label(yoe: float) -> str:
    if yoe is None:
        return "unknown"
    if 5 <= yoe <= 9:
        return f"{yoe:.0f} years — target band ✓"
    elif yoe < 5:
        return f"{yoe:.0f} years — below target (5–9 required)"
    else:
        return f"{yoe:.0f} years — above target (may be overqualified)"


def get_trust_label(trust_score: float) -> str:
    if trust_score is None:
        return "not assessed"
    if trust_score <= 0.1:
        return "clean profile"
    elif trust_score <= 0.3:
        return "minor concerns flagged"
    else:
        return "multiple risk flags — review manually"


def get_company_type_label(product_vs_services_score: float) -> str:
    if product_vs_services_score is None:
        return "unknown background"
    if product_vs_services_score >= 0.8:
        return "primarily product company"
    elif product_vs_services_score >= 0.5:
        return "mixed product/services"
    else:
        return "primarily IT services"
