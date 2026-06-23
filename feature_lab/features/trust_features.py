"""
trust_features.py — Trust score.

This module uses the real implementation from 04_trust_score_lab.md.
The output shape is fixed so the feature store schema doesn't change on drop-in.

Output dict shape:
  {
    "trust_score": float,        # 0.0–1.0 continuous (NEVER binary)
    "trust_reasons": list[str],  # list of reason strings
    "reliability_tag": str       # always "sparse" for the stub
  }
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, Any, Tuple
from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry
from trust_lab.scoring.ensemble_trust_score import compute_trust_score

class TrustScoreFeature(BaseFeature):
    """
    Real implementation calling 04_trust_score_lab.md logic.
    """
    def __init__(self):
        super().__init__("trust_score_composite", 1, "sparse")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        res = compute_trust_score(candidate)
        
        reasons = []
        for check, detail in res.get("details", {}).items():
            if detail and "No implausible claims" not in detail and "No assessment data available" not in detail:
                reasons.append(f"{check}: {detail}")
                
        val = {
            "trust_score": res["trust_score"],
            "trust_reasons": reasons,
        }
        return val, "sparse"

registry.register(TrustScoreFeature())
