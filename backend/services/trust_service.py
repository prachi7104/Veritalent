from trust_lab.scoring.ensemble_trust_score import compute_trust_score
from backend.config import get_trust_level
from backend.api.schemas.responses import (
    TrustBreakdown, TrustChecks, YoeTenureConsistencyCheck,
    ProficiencyPlausibilityCheck, KeywordStuffingDensityCheck,
    AssessmentCorroborationCheck, TemplateRelianceCheck
)
from trust_lab.checks.skill_density_check import DEEP_IR_SKILLS, BUZZWORD_SKILLS

def get_trust_breakdown(candidate_raw: dict) -> TrustBreakdown:
    res = compute_trust_score(candidate_raw)
    score = res["trust_score"]
    details = res["details"]
    breakdown = res["breakdown"]

    # YOE Risk
    yoe_det = details.get("yoe", {})
    yoe_flagged = yoe_det.get("flagged", False)
    yoe_dev = float(yoe_det.get("deviation_years", 0.0))
    if yoe_flagged:
        yoe_exp = f"Stated {yoe_det.get('stated_yoe', 0):.1f} years of experience does not match listed work history ({yoe_det.get('calculated_yoe', 0):.1f} years) — a deviation of {yoe_dev:.1f} years."
    else:
        yoe_exp = "YOE is consistent with career history."
    yoe_check = YoeTenureConsistencyCheck(
        score=float(breakdown.get("yoe_risk", 0.0)),
        flagged=yoe_flagged,
        explanation=yoe_exp,
        deviation_years=yoe_dev
    )

    # Proficiency Risk
    prof_det = details.get("prof", {})
    prof_flagged = prof_det.get("flagged", False)
    prof_sev = float(prof_det.get("severity_sum", 0.0))
    prof_count = len(prof_det.get("implausible_skills", []))
    if prof_flagged:
        prof_exp = f"Claimed 'expert' level on skills with low evidence (Severity sum: {prof_sev})."
    else:
        prof_exp = "Skill proficiencies are plausible."
    prof_check = ProficiencyPlausibilityCheck(
        score=float(breakdown.get("prof_risk", 0.0)),
        flagged=prof_flagged,
        explanation=prof_exp,
        implausible_skill_count=prof_count,
        severity_weighted_sum=prof_sev
    )

    # Assessment Corroboration
    assm_det = details.get("assm", {})
    assm_flagged = assm_det.get("flagged", False)
    if assm_flagged:
        assm_exp = "Failed to corroborate assessments."
    else:
        assm_exp = "Assessments generally corroborate skills."
    assm_check = AssessmentCorroborationCheck(
        score=float(breakdown.get("assm_risk", 0.0)),
        flagged=assm_flagged,
        explanation=assm_exp,
        coverage=float(assm_det.get("coverage", 0.0)),
        has_data=bool(assm_det.get("has_assessments", False))
    )

    # Template Reliance
    temp_det = details.get("temp", {})
    temp_flagged = temp_det.get("flagged", False)
    temp_frac = float(temp_det.get("template_fraction", 0.0))
    if temp_flagged:
        temp_exp = f"High template reliance: {temp_frac*100:.1f}% of descriptions match known boilerplate."
    else:
        temp_exp = "Original descriptions."
    temp_check = TemplateRelianceCheck(
        score=float(breakdown.get("temp_risk", 0.0)),
        flagged=temp_flagged,
        explanation=temp_exp,
        reliance_fraction=temp_frac,
        note="experimental signal — low weight"
    )

    # Keyword Stuffing Density (recalculating missing details)
    dens_det = details.get("dens", {})
    dens_score = float(breakdown.get("dens_risk", 0.0))
    dens_flagged = dens_score > 0.0
    
    career_timeline = candidate_raw.get("career_timeline", [])
    yoe = sum(1.0 for _ in career_timeline)
    profile_yoe = candidate_raw.get("profile", {}).get("years_of_experience", 0.0)
    years_experience = max(1.0, max(yoe, profile_yoe))
    skills = candidate_raw.get("skills", [])
    total_claimed = sum(1 for s in skills if s.get("name", "").lower() in DEEP_IR_SKILLS or s.get("name", "").lower() in BUZZWORD_SKILLS)
    
    if dens_flagged:
        dens_exp = "High density of keyword skills without corresponding experience."
    else:
        dens_exp = "Skill density within normal bounds."

    dens_check = KeywordStuffingDensityCheck(
        score=dens_score,
        flagged=dens_flagged,
        explanation=dens_exp,
        total_claimed_skills=total_claimed,
        years_experience=years_experience,
        density_ratio=float(dens_det.get("keyword_stuffing_density", 0.0))
    )

    checks = TrustChecks(
        yoe_tenure_consistency=yoe_check,
        proficiency_plausibility=prof_check,
        keyword_stuffing_density=dens_check,
        assessment_corroboration=assm_check,
        template_reliance=temp_check
    )

    caveat = "This trust score detects sloppy or inconsistent fraud only. It does NOT reliably catch sophisticated, internally-consistent fraud."
    level = get_trust_level(score)

    return TrustBreakdown(
        composite_score=score,
        level=level,
        checks=checks,
        caveat=caveat
    )
