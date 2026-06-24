import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.attribution.feature_contribution_summary import get_top_k_contributions

def test_shap_explainer():
    explainer = SHAPExplainer()
    # Dummy candidate
    candidate = {
        "skill_depth": 5.0,
        "experience_years": 8.0,
        "trust_score": 0.9
    }
    
    res = explainer.explain_candidate(candidate)
    
    assert "expected_value" in res
    assert "contributions" in res
    assert "skill_depth" in res["contributions"]
    
    # Test top-k extraction
    top_k = get_top_k_contributions(res, k=2)
    assert len(top_k) == 2
    assert "feature" in top_k[0]
    assert "shap_value" in top_k[0]
