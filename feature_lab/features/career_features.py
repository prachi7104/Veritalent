"""
career_features.py — Career trajectory and seniority progression features.

Ordinal title-rank lookup table (documented per task spec):
  1 = intern
  2 = junior, associate (entry-level)
  3 = engineer, developer, analyst (mid-level contributor)
  4 = senior (senior individual contributor)
  5 = lead (technical lead, team lead)
  6 = staff (staff engineer — broader scope than senior)
  7 = principal (cross-team technical authority)
  8 = head (head of engineering/AI/ML)
  9 = director
  10 = vp / vice president
  11 = cto, chief (C-suite)

Matching: longest matching token wins.
Candidates with no matching token default to rank 2 (entry-level).
"""
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry
from feature_lab.features.skill_features import get_skill_band

# ---------------------------------------------------------------------------
# Title rank lookup — ordered so longer/more-specific strings match first
# ---------------------------------------------------------------------------
TITLE_RANKS = [
    ("vice president", 10),
    ("principal", 7),
    ("staff", 6),
    ("director", 9),
    ("head", 8),
    ("chief", 11),
    ("senior", 4),
    ("lead", 5),
    ("vp", 10),
    ("cto", 11),
    ("associate", 2),
    ("junior", 2),
    ("intern", 1),
    ("engineer", 3),
    ("developer", 3),
    ("analyst", 3),
    ("scientist", 3),
    ("researcher", 3),
    ("architect", 6),
    ("manager", 5),
]


def get_title_rank(title: str) -> int:
    """Return ordinal rank for a job title string."""
    t = title.lower()
    for token, rank in TITLE_RANKS:
        if token in t:
            return rank
    return 2  # Default: entry-level if unknown


class CareerVelocityFeature(BaseFeature):
    """
    YOE / distinct employers — higher value = fewer employers per year = more stability.
    Inverted so stability reads as a higher score.
    reliability: clean
    """
    def __init__(self):
        super().__init__("career_velocity", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        yoe = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0)
        career = candidate.get("career_history", [])
        if not career or yoe == 0:
            return 0.0, self.default_reliability_tag

        distinct_employers = len({r.get("company", "") for r in career if r.get("company")})
        if distinct_employers == 0:
            return 0.0, self.default_reliability_tag

        velocity = yoe / distinct_employers
        return velocity, self.default_reliability_tag


class PromotionVelocityFeature(BaseFeature):
    """
    (max_rank − min_rank) / YOE — rate of title-level increase over career.
    Uses ordinal rank table above.
    reliability: clean
    """
    def __init__(self):
        super().__init__("promotion_velocity", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        career = candidate.get("career_history", [])
        if len(career) <= 1:
            return 0.0, self.default_reliability_tag

        ranks = [get_title_rank(r.get("title", "")) for r in career]
        max_rank = max(ranks)
        min_rank = min(ranks)

        yoe = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0)
        if yoe <= 0:
            return 0.0, self.default_reliability_tag

        rate = (max_rank - min_rank) / yoe
        return rate, self.default_reliability_tag


class TenureStabilityFeature(BaseFeature):
    """
    Average duration_months per role. Higher = more stable/tenured.
    reliability: clean
    """
    def __init__(self):
        super().__init__("tenure_stability", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        career = candidate.get("career_history", [])
        if not career:
            return 0.0, self.default_reliability_tag

        durations = [float(r.get("duration_months", 0) or 0) for r in career]
        avg_dur = sum(durations) / len(durations)
        return avg_dur, self.default_reliability_tag


class InflectionPointStrengthFeature(BaseFeature):
    """
    Largest single role-transition jump in the candidate's career,
    measured by title-rank delta + industry change (binary 0/1 bonus).
    Moderately speculative — tagged 'experimental' per architecture review.
    reliability: experimental
    """
    def __init__(self):
        super().__init__("inflection_point_strength", 1, "experimental")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        career = candidate.get("career_history", [])
        if len(career) <= 1:
            return 0.0, self.default_reliability_tag

        max_jump = 0.0
        # career_history is typically newest-first; compare adjacent entries
        for i in range(len(career) - 1):
            curr_role = career[i]
            prev_role = career[i + 1]

            r1 = get_title_rank(curr_role.get("title", ""))
            r2 = get_title_rank(prev_role.get("title", ""))
            jump = float(max(0, r1 - r2))

            ind1 = curr_role.get("industry", "")
            ind2 = prev_role.get("industry", "")
            if ind1 and ind2 and ind1 != ind2:
                jump += 1.0

            if jump > max_jump:
                max_jump = jump

        return max_jump, self.default_reliability_tag


# ---------------------------------------------------------------------------
# Register all features
# ---------------------------------------------------------------------------
registry.register(CareerVelocityFeature())
registry.register(PromotionVelocityFeature())
registry.register(TenureStabilityFeature())
registry.register(InflectionPointStrengthFeature())
