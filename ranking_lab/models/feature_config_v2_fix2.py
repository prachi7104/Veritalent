# ranking_lab/models/feature_config_v2_fix2.py
"""
v2 Fix-2: Replace activity_quality_composite with recruiter_response_rate.
"""

FEATURE_NAMES_FIX2 = [
    "skill_depth",
    "skill_breadth",
    "skill_mastery_triangulation",
    "jd_skill_score",
    "tenure_stability",
    "recruiter_response_rate",   # Replaces activity_quality_composite
    "trust_score",
    "logistics_fit_score",
    "product_vs_services",
    "yoe_band_fit",
]

MONOTONIC_CONSTRAINTS_FIX2 = [
    1,   # skill_depth
    0,   # skill_breadth
    0,   # skill_mastery_triangulation
    1,   # jd_skill_score
    0,   # tenure_stability
    1,   # recruiter_response_rate
    -1,  # trust_score
    0,   # logistics_fit_score
    0,   # product_vs_services
    1,   # yoe_band_fit
]

assert len(FEATURE_NAMES_FIX2) == len(MONOTONIC_CONSTRAINTS_FIX2)
