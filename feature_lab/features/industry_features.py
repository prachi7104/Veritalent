"""
industry_features.py — Industry relevance mapping.

JD-relevance tier table (explicit, per task spec):
  Tier 3 (most relevant): AI/ML, Software, SaaS, Technology
  Tier 2 (moderate):      Fintech, E-commerce, EdTech, HealthTech, Gaming
  Tier 1 (low):           IT Services, Conglomerate, Manufacturing, Logistics
  Tier 0 (unrelated):     Paper Products, Insurance Tech, Real Estate,
                          Agriculture, and any unmapped category

Scoring: max(historical industry tiers) / 3.0 → normalized 0.0–1.0
"""
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry

INDUSTRY_TIERS: Dict[str, int] = {
    # Tier 3
    "ai": 3, "ml": 3, "machine learning": 3, "software": 3, "saas": 3,
    "technology": 3, "tech": 3, "information technology": 3,
    # Tier 2
    "fintech": 2, "e-commerce": 2, "ecommerce": 2, "edtech": 2,
    "healthtech": 2, "health tech": 2, "gaming": 2, "media": 2,
    "internet": 2, "startup": 2,
    # Tier 1
    "it services": 1, "conglomerate": 1, "manufacturing": 1,
    "logistics": 1, "retail": 1, "telecom": 1, "bfsi": 1,
    # Tier 0 — unrelated
    "paper": 0, "insurance": 0, "real estate": 0, "agriculture": 0,
}


def get_industry_score(industry: str) -> int:
    """Return relevance tier for an industry string."""
    if not industry:
        return 0
    ind = industry.lower()
    best = 0
    for keyword, tier in INDUSTRY_TIERS.items():
        if keyword in ind and tier > best:
            best = tier
    return best


class IndustryRelevanceFeature(BaseFeature):
    """
    Max relevance tier across current + historical industries, normalized to 0–1.
    reliability: clean
    """
    def __init__(self):
        super().__init__("industry_relevance", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        max_score = 0

        curr_industry = candidate.get("profile", {}).get("current_industry", "")
        max_score = max(max_score, get_industry_score(curr_industry))

        for role in candidate.get("career_history", []):
            ind = role.get("industry", "")
            max_score = max(max_score, get_industry_score(ind))

        return max_score / 3.0, self.default_reliability_tag


registry.register(IndustryRelevanceFeature())
