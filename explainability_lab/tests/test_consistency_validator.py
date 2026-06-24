import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from explainability_lab.narrative.consistency_validator import validate_consistency
from explainability_lab.narrative.fallback_narrative import generate_fallback
from explainability_lab.attribution.shap_explainer import SHAPExplainer

def test_consistency_validator_pass():
    shap_summary = [
        {"feature": "skill_depth", "raw_value": 4.0, "shap_value": 1.2},
        {"feature": "experience_years", "raw_value": 5.0, "shap_value": 0.5}
    ]
    # Exact mention of the features as per Option A
    narrative = "The candidate scored well because of their **skill_depth** and their **experience_years**."
    assert validate_consistency(narrative, shap_summary) == True

def test_consistency_validator_fail_hallucination():
    shap_summary = [
        {"feature": "skill_depth", "raw_value": 4.0, "shap_value": 1.2}
    ]
    # Mentions a valid feature but ALSO mentions one that isn't in the top-k list
    narrative = "High **skill_depth** and excellent **skill_breadth**."
    assert validate_consistency(narrative, shap_summary) == False

def test_consistency_validator_fail_no_exact_match():
    shap_summary = [
        {"feature": "skill_depth", "raw_value": 4.0, "shap_value": 1.2}
    ]
    # Fails to mention the exact bold string
    narrative = "High depth of skills."
    assert validate_consistency(narrative, shap_summary) == False
