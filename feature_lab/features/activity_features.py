"""
activity_features.py — Candidate activity and platform-engagement features.

Critical constraint (per schema guard and master context):
  github_activity_score == -1 is MISSING data, NOT a zero score.
  It must be excluded from the composite entirely (contributes zero weight).
"""
from typing import Dict, Any, Tuple
from datetime import date, datetime

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry

# Reference date: use a sentinel that is computed lazily from the data.
# We default to today, making recency relative to when the store is generated.
def _today() -> date:
    return date.today()


def _parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00").split("T")[0]).date()
    except Exception:
        return None


class ActivityQualityCompositeFeature(BaseFeature):
    """
    Composite activity score combining:
      - last_active_date recency (decayed, not a hard cutoff)
      - recruiter_response_rate
      - interview_completion_rate
      - github_activity_score  (CRITICAL: -1 sentinel treated as missing,
                                not as zero — it contributes zero weight)
    """

    def __init__(self):
        super().__init__("activity_quality_composite", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        signals = candidate.get("redrob_signals", {})
        today = _today()

        weighted_sum = 0.0
        total_weight = 0.0

        # 1. Last-active recency — decay over 180 days (6 months)
        last_active_str = signals.get("last_active_date")
        last_active = _parse_date(last_active_str)
        if last_active is not None:
            days_inactive = (today - last_active).days
            # Clamp: 1.0 if active today, 0.0 at 180+ days
            recency_score = max(0.0, 1.0 - (days_inactive / 180.0))
        else:
            recency_score = 0.5  # Unknown → conservative midpoint

        weighted_sum += recency_score * 1.0
        total_weight += 1.0

        # 2. Recruiter response rate (0.0–1.0)
        rrr = signals.get("recruiter_response_rate")
        rrr_score = float(rrr) if rrr is not None else 0.5
        weighted_sum += rrr_score * 1.0
        total_weight += 1.0

        # 3. Interview completion rate
        icr = signals.get("interview_completion_rate")
        icr_score = float(icr) if icr is not None else 0.5
        weighted_sum += icr_score * 1.0
        total_weight += 1.0

        # 4. GitHub activity score — -1 is MISSING, not zero
        github = signals.get("github_activity_score")
        if github is not None and float(github) != -1.0:
            gh_score = float(github) / 100.0
            weighted_sum += gh_score * 1.0
            total_weight += 1.0
        # else: do NOT add any weight — missing data contributes nothing

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        return composite, self.default_reliability_tag


registry.register(ActivityQualityCompositeFeature())
