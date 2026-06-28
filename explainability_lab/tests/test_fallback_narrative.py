import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from explainability_lab.narrative.fallback_narrative import generate_fallback_narrative

def test_fallback_contains_no_raw_feature_names():
    """Fallback narrative must not expose raw feature names."""
    context = {
        "rank": 5,
        "current_title": "ML Engineer",
        "current_company": "Flipkart",
        "company_type": "primarily product company",
        "yoe": "7",
        "yoe_band_label": "7 years — target band ✓",
        "jd_skills": "Elasticsearch, PyTorch",
        "shap_summary": "JD-aligned skill strength (+4.2, primary driver)",
        "trust_label": "clean profile",
        "notice_period": "30 days",
    }
    text = generate_fallback_narrative(context)
    raw_features = ["skill_mastery_triangulation", "skill_depth",
                    "activity_quality_composite"]
    for feat in raw_features:
        assert feat not in text, f"Fallback contains raw feature name: {feat}"


def test_shap_formatter_translates_feature_names():
    """SHAP formatter must translate raw names to English."""
    from explainability_lab.narrative.shap_formatter import format_shap_for_narrative
    raw_shap = [
        {"feature": "skill_mastery_triangulation", "shap_value": 4.5},
        {"feature": "logistics_fit_score", "shap_value": 1.2},
        {"feature": "trust_score", "shap_value": -0.3},
    ]
    result = format_shap_for_narrative(raw_shap)
    assert "skill_mastery_triangulation" not in result
    assert "verified skill depth" in result
    assert "location and availability fit" in result
    assert "concern" in result  # trust negative should be labeled concern
