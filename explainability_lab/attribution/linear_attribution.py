import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

LINEAR_WEIGHTS = {
    'skill_depth': 2.5,
    'experience_years': 1.0,
    'tenure_stability': 0.8,
    'education_level': 0.5,
    'leadership_signal': 1.2,
    'trust_score': 3.0,
    'endorsement_count': 0.2
}

def explain_linear(candidate: dict) -> dict:
    """
    Trivial attribution for linear model: coefficient * value.
    Provides a baseline for sanity-checking SHAP.
    """
    contributions = {}
    base_score = 0.0
    
    for f in TRAINING_FEATURES:
        val = float(candidate.get(f, 0.0) or 0.0)
        weight = LINEAR_WEIGHTS.get(f, 0.0)
        contrib = val * weight
        contributions[f] = {
            "raw_value": val,
            "shap_value": contrib # For consistency of shape
        }
        
    return {
        "expected_value": 0.0,
        "contributions": contributions
    }
