import os
import sys
import numpy as np
import shap
import lightgbm as lgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

class SHAPExplainer:
    def __init__(self, model_path="ranking_lab/models/gbm_lambdarank.txt"):
        """
        Loads the LightGBM booster directly and wraps it in a SHAP TreeExplainer.
        """
        self.booster = lgb.Booster(model_file=model_path)
        self.explainer = shap.TreeExplainer(self.booster)
        self.features = TRAINING_FEATURES

    def explain_candidate(self, candidate: dict) -> dict:
        """
        Returns raw SHAP values for the given candidate.
        """
        row = []
        for f in self.features:
            row.append(float(candidate.get(f, 0.0) or 0.0))
        X = np.array([row])
        
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list): # For multiclass, but lambdarank is 1D
            shap_values = shap_values[1]
            
        shap_vals = shap_values[0] # Single candidate
        
        contributions = {}
        for i, f in enumerate(self.features):
            contributions[f] = {
                "raw_value": row[i],
                "shap_value": float(shap_vals[i])
            }
            
        expected_value = float(self.explainer.expected_value)
        if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
            
        return {
            "expected_value": expected_value,
            "contributions": contributions
        }
