import os
import sys

# Add root to path so imports work when running from any dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.checks.yoe_tenure_consistency import check_yoe_consistency
from trust_lab.checks.proficiency_plausibility import check_proficiency_plausibility
from trust_lab.checks.assessment_corroboration import check_assessment_corroboration
from trust_lab.checks.template_reliance_signal import check_template_reliance
from trust_lab.checks.identity_verification import check_identity_verification

def compute_trust_score(candidate: dict, known_templates: set = None, weights: dict = None) -> dict:
    """
    Combines all checks into a single calibrated probability (0.0-1.0),
    with the per-check contribution exposed.
    NEVER implement an auto-reject threshold anywhere in this module.
    """
    if weights is None:
        weights = {
            "yoe_consistency": 1.0,
            "proficiency": 1.0,
            "assessment": 1.0,
            "template": 0.2, # configured to default low weight
            "identity": 0.5
        }
        
    yoe_res = check_yoe_consistency(candidate)
    prof_res = check_proficiency_plausibility(candidate)
    assm_res = check_assessment_corroboration(candidate)
    temp_res = check_template_reliance(candidate, known_templates)
    ident_res = check_identity_verification(candidate)
    
    # Cap individual risks at 1.0
    # YOE deviation: we start risk accumulation above 1.5 years
    yoe_risk = min((yoe_res["deviation_years"] - 1.5) / 3.0, 1.0) if yoe_res["flagged"] else 0.0
    prof_risk = min(prof_res["severity_sum"] / 2.0, 1.0)
    
    assm_risk = 0.0
    if assm_res["has_assessments"] and assm_res["average_delta"] > 0:
        assm_risk = min(assm_res["average_delta"] / 40.0, 1.0)
        
    temp_risk = temp_res["template_fraction"]
    ident_risk = ident_res["unverified_fraction"]
    
    total_weight = weights["yoe_consistency"] + weights["proficiency"] + \
                   (weights["assessment"] * assm_res["weight_multiplier"]) + \
                   weights["template"] + weights["identity"]
                   
    weighted_sum = (
        yoe_risk * weights["yoe_consistency"] +
        prof_risk * weights["proficiency"] +
        assm_risk * weights["assessment"] * assm_res["weight_multiplier"] +
        temp_risk * weights["template"] +
        ident_risk * weights["identity"]
    )
    
    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    return {
        "trust_score": float(final_score),
        "breakdown": {
            "yoe_risk": float(yoe_risk),
            "prof_risk": float(prof_risk),
            "assm_risk": float(assm_risk),
            "temp_risk": float(temp_risk),
            "ident_risk": float(ident_risk)
        },
        "details": {
            "yoe": yoe_res["details"],
            "prof": prof_res["details"],
            "assm": assm_res["details"],
            "temp": temp_res["details"],
            "ident": ident_res["details"]
        }
    }
