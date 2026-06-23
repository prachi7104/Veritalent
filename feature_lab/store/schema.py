"""
schema.py — Explicit schema for the feature store output.

One row per candidate. Columns:
  - candidate_id: str
  - <feature_name>: typed value (float, list, etc.)
  - <feature_name>_reliability: str — one of 'clean', 'sparse', 'experimental'

GBM_INPUT_FEATURES and DISPLAY_ONLY_FEATURES are consumed by Lab 06's
training script to determine which columns to include in the training DataFrame.
Do NOT modify these lists without updating the ablation report and LAB06_NOTES.md.
"""
from typing import Dict

# ---------------------------------------------------------------------------
# Full schema (all feature store columns)
# ---------------------------------------------------------------------------
FEATURE_SCHEMA: Dict[str, type] = {
    "candidate_id": str,

    "skill_depth": float,
    "skill_depth_reliability": str,
    "skill_breadth": float,
    "skill_breadth_reliability": str,
    "skill_recency": float,
    "skill_recency_reliability": str,
    "skill_mastery_triangulation": float,
    "skill_mastery_triangulation_reliability": str,

    "career_velocity": float,
    "career_velocity_reliability": str,
    "promotion_velocity": float,
    "promotion_velocity_reliability": str,
    "tenure_stability": float,
    "tenure_stability_reliability": str,
    "inflection_point_strength": float,
    "inflection_point_strength_reliability": str,

    "trust_score": float,
    "trust_reasons": list,
    "trust_score_composite_reliability": str,

    "activity_quality_composite": float,
    "activity_quality_composite_reliability": str,

    "industry_relevance": float,
    "industry_relevance_reliability": str,

    "logistics_fit_score": float,
    "logistics_fit_score_reliability": str,

    "product_vs_services": float,
    "product_vs_services_reliability": str,
}

# ---------------------------------------------------------------------------
# Lab 06 feature split — determined by ablation (see reports/LAB06_NOTES.md)
# ---------------------------------------------------------------------------

# Features to include as GBM training inputs.
# NOTE: industry_relevance is intentionally absent — ablation proved it adds
# noise. Let the GBM learn (or not learn) it only if you include it explicitly.
GBM_INPUT_FEATURES = [
    "skill_depth",
    "skill_breadth",
    "skill_recency",
    "skill_mastery_triangulation",
    "tenure_stability",           # use instead of career_velocity (corr=0.75)
    "promotion_velocity",
    "inflection_point_strength",  # tagged experimental — verify SHAP contribution
    "trust_score",                # stub until Lab 04; monotonic increasing constraint
    "activity_quality_composite",
    "logistics_fit_score",
    "product_vs_services",
]

# Features to expose in recruiter narratives (Lab 07) but NOT fed to GBM.
# career_velocity: correlated with tenure_stability (0.75); reads naturally
#   in human explanations ("averaged X years per employer") but redundant for model.
# industry_relevance: ablation shows +0.0168 NDCG when removed — noise, not signal.
#   Keep in store for explainability, do not pre-weight in GBM.
DISPLAY_ONLY_FEATURES = [
    "career_velocity",
    "industry_relevance",
]


def validate_schema(record: dict) -> bool:
    """Verify that a record conforms to the defined schema."""
    for key, expected_type in FEATURE_SCHEMA.items():
        if key not in record:
            return False
        if record[key] is not None and not isinstance(record[key], expected_type):
            # Allow int where float is expected
            if expected_type == float and isinstance(record[key], int):
                continue
            return False
    return True
