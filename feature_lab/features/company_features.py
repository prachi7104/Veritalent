"""
company_features.py — Product-vs-services classification.

Known services firms (conservative list — prefer false negatives over false positives):
  TCS, Infosys, Wipro, Cognizant, Accenture, Capgemini,
  HCL, Tech Mahindra, Mindtree

Classification scores (0.0 = pure services, 1.0 = pure product):
  1.0 — career-long product only (no services firm ever)
  0.9 — current role is product; prior services experience exists
  0.8 — current role is services; prior product experience exists
        (JD explicitly does NOT penalise this — "carve-out" from master context)
  0.0 — career-long services only

Any company not in the services list is treated as a product company.
Requires a specific passing test (see tests/test_features.py).
"""
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry

SERVICES_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "cognizant",
    "accenture", "capgemini", "hcl", "hcl technologies", "tech mahindra", "mindtree",
}


def is_services(company_name: str) -> bool:
    if not company_name:
        return False
    name = company_name.lower()
    return any(sf in name for sf in SERVICES_FIRMS)


class ProductVsServicesClassificationFeature(BaseFeature):
    """
    Sequence-aware product-vs-services score.
    Iterates the FULL career_history, not just current_company.
    reliability: clean
    """
    def __init__(self):
        super().__init__("product_vs_services", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        career = candidate.get("career_history", [])
        if not career:
            return 0.5, self.default_reliability_tag

        has_product = False
        has_services = False
        current_is_services = False

        for role in career:
            company = role.get("company", "")
            if role.get("is_current", False):
                if is_services(company):
                    current_is_services = True
                    has_services = True
                else:
                    has_product = True
            else:
                if is_services(company):
                    has_services = True
                else:
                    has_product = True

        # Scoring matrix
        if not has_services:
            return 1.0, self.default_reliability_tag
        elif has_services and not current_is_services and has_product:
            # Current is product, prior services
            return 0.9, self.default_reliability_tag
        elif current_is_services and has_product:
            # Current is services, prior product — JD carve-out
            return 0.8, self.default_reliability_tag
        else:
            # Services only throughout career
            return 0.0, self.default_reliability_tag


registry.register(ProductVsServicesClassificationFeature())
