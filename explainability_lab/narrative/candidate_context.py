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


def build_blend_annotation(features: dict, pool_jd_mean: float,
                            pool_yoe_mean: float) -> dict:
    """
    Adds jd_skill_score and yoe_band_fit as explicit context for the narrative,
    since these features do not appear in SHAP (they are score-level adjustments,
    not GBM features).

    Returns a dict with human-readable labels for the narrative prompt.
    """
    jd_score  = float(features.get("jd_skill_score",  0) or 0)
    yoe_score = float(features.get("yoe_band_fit",    0) or 0)

    # JD skill label
    if jd_score > pool_jd_mean * 1.5:
        jd_label = f"significantly above pool average ({jd_score:.0f} vs avg {pool_jd_mean:.0f}) — strong JD alignment"
    elif jd_score > pool_jd_mean:
        jd_label = f"above pool average ({jd_score:.0f} vs avg {pool_jd_mean:.0f})"
    elif jd_score > 0:
        jd_label = f"below pool average ({jd_score:.0f} vs avg {pool_jd_mean:.0f})"
    else:
        jd_label = "no JD-relevant skills detected"

    # YOE band label
    yoe_band_labels = {
        1.00: "target band ✓ (5–9 years)",
        0.75: "slightly below target (4–5 years)",
        0.70: "slightly above target (9–12 years)",
        0.40: "outside target band",
        0.10: "significantly junior",
        0.25: "significantly senior",
        0.50: "experience unknown",
    }
    yoe_label = yoe_band_labels.get(round(yoe_score, 2), f"score={yoe_score:.2f}")

    return {
        "jd_skill_label": jd_label,
        "yoe_band_label_text": yoe_label,
        "jd_score_raw": round(jd_score, 1),
        "yoe_score_raw": round(yoe_score, 2),
    }


def build_candidate_context(
    candidate: dict,
    features: dict,
    shap_contributions: list[dict],
    rank: int,
    pool_jd_skill_mean: float,
    pool_yoe_mean: float = 0.64,
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

    blend_ann = build_blend_annotation(features, pool_jd_skill_mean, pool_yoe_mean)

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
        # blend annotation fields
        "jd_skill_label": blend_ann["jd_skill_label"],
        "yoe_band_label_text": blend_ann["yoe_band_label_text"],
        "trust_label": get_trust_label(features.get("trust_score")),
    }

