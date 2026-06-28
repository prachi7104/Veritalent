"""
Feature configuration v2.
12 features. Every constraint is justified by Phase 1 evidence.

Constraint rationale:
  skill_depth (+1):              More IR skill coverage → better.
  skill_breadth (0):             Band coverage can penalise specialists. Unconstrained.
  skill_recency (+1):            More recent IR skills → better. v2 now working.
  skill_mastery_triangulation(0):Keep for comparison but unconstrained.
                                  Ablation (PROMPT_03A) decides if redundant.
  jd_skill_score (+1):           Higher JD-weighted skill → better. Core new signal.
  tenure_stability (0):          Long tenures ≠ better. Unconstrained.
  activity_quality_composite (0):Complex composite — let GBM decide direction.
  trust_score (-1):              Higher trust_score = higher risk. Monotonic penalise.
  logistics_fit_score (0):       Nonlinear geography. Unconstrained.
  product_vs_services (0):       Mixed careers vary. Unconstrained.
  implied_skill_score (+1):      IR terms in summary → signal. 98.9% zero but
                                  the 1.1% that fire are top candidates.
  yoe_band_fit (+1):             Higher band fit → better. Clean monotonic.
"""

FEATURE_NAMES_V2 = [
    "skill_depth",
    "skill_breadth",
    "skill_recency",
    "skill_mastery_triangulation",
    "jd_skill_score",
    "tenure_stability",
    "activity_quality_composite",
    "trust_score",
    "logistics_fit_score",
    "product_vs_services",
    "implied_skill_score",
    "yoe_band_fit",
]

MONOTONIC_CONSTRAINTS_V2 = [
    1,   # skill_depth
    0,   # skill_breadth
    1,   # skill_recency
    0,   # skill_mastery_triangulation
    1,   # jd_skill_score
    0,   # tenure_stability
    0,   # activity_quality_composite
    -1,  # trust_score
    0,   # logistics_fit_score
    0,   # product_vs_services
    1,   # implied_skill_score
    1,   # yoe_band_fit
]

assert len(FEATURE_NAMES_V2) == len(MONOTONIC_CONSTRAINTS_V2)
