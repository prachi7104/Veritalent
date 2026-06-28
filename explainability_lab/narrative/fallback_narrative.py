def generate_fallback_narrative(context: dict) -> str:
    """
    Structured fallback when LLM is unavailable.
    Uses the same context dict as the LLM prompt.
    Produces recruiter-readable text, not feature names.
    """
    yoe_label = context.get("yoe_band_label", "unknown experience level")
    company = context.get("current_company", "unknown company")
    company_type = context.get("company_type", "unknown background")
    title = context.get("current_title", "unknown role")
    skills = context.get("jd_skills", "not specified")
    shap = context.get("shap_summary", "")
    trust = context.get("trust_label", "not assessed")
    rank = context.get("rank", "?")
    notice = context.get("notice_period", "unknown")

    return (
        f"Ranked #{rank}. Currently {title} at {company} ({company_type}), "
        f"with {yoe_label}. "
        f"JD-relevant skills include: {skills}. "
        f"Key ranking signals: {shap}. "
        f"Availability: {notice} notice. "
        f"Profile credibility: {trust}."
    )
