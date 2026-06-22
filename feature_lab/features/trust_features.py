"""
trust_features.py — Trust score stub.

This module will be replaced with the real implementation from 04_trust_score_lab.md.
The output shape is fixed so the feature store schema doesn't change on drop-in.

Output dict shape:
  {
    "trust_score": float,        # 0.0–1.0 continuous (NEVER binary)
    "trust_reasons": list[str],  # list of reason strings
    "reliability_tag": str       # always "sparse" for the stub
  }
"""
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry


class TrustScoreFeature(BaseFeature):
    """
    Stub implementation. Returns a neutral 0.5 trust score.
    Replace with real implementation from 04_trust_score_lab.md.
    """
    def __init__(self):
        super().__init__("trust_score_composite", 1, "sparse")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        val = {
            "trust_score": 0.5,
            "trust_reasons": [],
        }
        return val, "sparse"


registry.register(TrustScoreFeature())
