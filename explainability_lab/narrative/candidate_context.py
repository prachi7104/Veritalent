"""
Builds the human-readable context dict passed to the narrative prompt.
Pulls from candidate profile + feature store.
"""
from explainability_lab.narrative.shap_formatter import (
    format_shap_for_narrative,
    get_yoe_band_label,
    get_trust_label,
    get_company_type_label,
)


def build_candidate_context(
    candidate: dict,
    features: dict,
    shap_contributions: list[dict],
    rank: int,
    pool_jd_skill_mean: float,
) -> dict:
    """Build the template context for the narrative LLM prompt."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Current role
    current_role = next(
        (r for r in career if r.get("is_current")), {}
    ) if career else {}
    current_title = (
        current_role.get("title") or profile.get("headline") or "Unknown title"
    )
    current_company = current_role.get("company") or "Unknown company"

    # YOE
    yoe = float(profile.get("years_of_experience", 0) or 0)

    # Location
    location = profile.get("location") or "Unknown location"

    # Notice period
    notice = features.get("logistics_fit_score_notice_period_days")
    if notice is not None:
        notice_label = (
            f"immediate" if float(notice) <= 15
            else f"{int(float(notice))} days"
        )
    else:
        notice_label = "unknown"

    # JD-relevant skills (deep-IR and fingerprint only)
    try:
        from feature_lab.features.skill_features import get_skill_band
        jd_skills = [
            s["name"] for s in skills
            if get_skill_band(s.get("name", "")) in ("deep-ir", "fingerprint")
        ][:6]
    except ImportError:
        jd_skills = [s["name"] for s in skills[:6]]
        
    jd_skills_str = ", ".join(jd_skills) if jd_skills else "none explicitly listed"

    return {
        "rank": rank,
        "current_title": current_title,
        "yoe": f"{yoe:.0f}" if yoe > 0 else "unknown",
        "current_company": current_company,
        "company_type": get_company_type_label(features.get("product_vs_services")),
        "location": location,
        "notice_period": notice_label,
        "jd_skills": jd_skills_str,
        "shap_summary": format_shap_for_narrative(shap_contributions),
        "jd_skill_score": float(features.get("jd_skill_score", 0) or 0),
        "pool_jd_skill_mean": pool_jd_skill_mean,
        "yoe_band_label": get_yoe_band_label(yoe),
        "trust_label": get_trust_label(features.get("trust_score")),
    }
