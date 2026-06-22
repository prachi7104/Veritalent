"""
logistics_features.py — Logistics fit: notice period, location, work-mode.

Notice period decay:
  - JD prefers sub-30-day notice. NOT a hard cutoff.
  - Uses exponential decay: score(np) = exp(-(np - 30) / 30) for np > 30
  - score(np <= 30) = 1.0

Location scoring:
  - Pune / Noida = 1.0 (JD target)
  - Tier-1 India cities = 0.8
  - Other India = 0.5
  - Non-India = 0.3 (NOT 0 — JD states "case-by-case", never auto-excluded)
  - willing_to_relocate adds +0.2 (capped at 1.0)
"""
import math
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry

TIER1_INDIA = {"bangalore", "bengaluru", "hyderabad", "mumbai", "delhi",
               "chennai", "kolkata", "ahmedabad", "gurugram", "gurgaon"}


class LogisticsFitScoreFeature(BaseFeature):
    """
    Composite logistics fit score (0.0–1.0).
    Components: notice_period decay + work_mode match + location fit.
    reliability: clean
    """
    def __init__(self):
        super().__init__("logistics_fit_score", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        signals = candidate.get("redrob_signals", {})
        profile = candidate.get("profile", {})

        # 1. Notice period soft decay
        np_days = signals.get("notice_period_days")
        if np_days is not None:
            np_days = float(np_days)
            np_score = 1.0 if np_days <= 30 else math.exp(-(np_days - 30) / 30.0)
        else:
            np_score = 0.5  # unknown → midpoint

        # 2. Preferred work mode
        pwm = (signals.get("preferred_work_mode") or "").lower()
        if "hybrid" in pwm:
            mode_score = 1.0
        elif "remote" in pwm:
            mode_score = 0.7
        elif "onsite" in pwm or "office" in pwm:
            mode_score = 0.8
        else:
            mode_score = 0.5

        # 3. Location fit
        loc = (profile.get("location") or "").lower()
        country = (profile.get("country") or "").lower()

        if "pune" in loc or "noida" in loc:
            loc_score = 1.0
        elif any(city in loc for city in TIER1_INDIA):
            loc_score = 0.8
        elif "india" in country or "india" in loc:
            loc_score = 0.5
        else:
            loc_score = 0.3  # Non-India: soft modifier only, never zero

        # Relocation bonus
        wtr = signals.get("willing_to_relocate", False)
        if wtr and loc_score < 1.0:
            loc_score = min(1.0, loc_score + 0.2)

        composite = (np_score + mode_score + loc_score) / 3.0
        return composite, self.default_reliability_tag


registry.register(LogisticsFitScoreFeature())
