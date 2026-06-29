# ranking_lab/models/feature_config_v2_fix1.py
"""
v2 Fix-1: 10-feature set.
Removed: skill_recency (importance 0.0041), implied_skill_score (98.9% zero, overfits)
Added constraint: activity_quality_composite = +1 (evidence: higher engagement → better)
"""

FEATURE_NAMES_FIX1 = [
    "skill_depth",
    "skill_breadth",
    "skill_mastery_triangulation",
    "jd_skill_score",
    "tenure_stability",
    "activity_quality_composite",
    "trust_score",
    "logistics_fit_score",
    "product_vs_services",
    "yoe_band_fit",
]

MONOTONIC_CONSTRAINTS_FIX1 = [
    1,   # skill_depth
    0,   # skill_breadth
    0,   # skill_mastery_triangulation
    1,   # jd_skill_score
    0,   # tenure_stability
    1,   # activity_quality_composite  # CHANGED from 0 to +1
    -1,  # trust_score
    0,   # logistics_fit_score
    0,   # product_vs_services
    1,   # yoe_band_fit
]

assert len(FEATURE_NAMES_FIX1) == len(MONOTONIC_CONSTRAINTS_FIX1)
