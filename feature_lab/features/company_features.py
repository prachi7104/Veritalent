"""
company_features.py — Product-vs-services classification.
"""
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry

SERVICES_FIRMS = frozenset({
    # Tier 1 — Large global IT services (original 9)
    "tata consultancy services", "tcs",
    "infosys",
    "wipro",
    "cognizant", "cognizant technology solutions",
    "accenture",
    "capgemini",
    "hcl technologies", "hcl",
    "tech mahindra",
    "mindtree",

    # Tier 2 — Large Indian IT services (newly added)
    "mphasis",
    "hexaware", "hexaware technologies",
    "niit technologies", "niit",
    "persistent systems",
    "l&t infotech", "larsen & toubro infotech", "lt infotech", "lti",
    "ltimindtree",  # post-merger entity
    "mastech",
    "zensar", "zensar technologies",
    "birlasoft",
    "cyient",
    "sonata software",
    "mphasis bfl",
    "patni computer systems",
    "rolta",
    "infotech enterprises",
    "kpit technologies",

    # Global services arms commonly found in Indian profiles
    "ibm global services", "ibm india",
    "ey", "ernst & young", "ey gds",
    "deloitte", "deloitte india",
    "kpmg", "kpmg india",
    "pwc", "pricewaterhousecoopers",
    "dxc technology",
    "unisys",
    "cgi",
    "ntt data",
    "fujitsu",

    # BPO/KPO often miscategorized
    "wns", "wns global services",
    "genpact",
    "hexaware bps",
    "mphasis bps",
    "syntel",  # now atos syntel
    "atos", "atos syntel",
})


def _normalize_company_name(name: str) -> str:
    """Lowercase, strip common suffixes for fuzzy matching."""
    import re
    name = name.lower().strip()
    # Remove common suffixes that vary across profiles
    name = re.sub(
        r'\b(pvt\.?|ltd\.?|limited|private|inc\.?|llc|llp|'
        r'technologies|technology|solutions|systems|services|'
        r'global|india|international|consulting)\b',
        '',
        name
    ).strip()
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _is_services_firm(company_name: str) -> bool:
    """
    Returns True if the company is a known services/outsourcing firm.
    Uses exact match first, then normalized match, then word/token match.
    """
    if not company_name:
        return False

    import re
    raw = company_name.lower().strip()
    if raw in SERVICES_FIRMS:
        return True

    normalized = _normalize_company_name(company_name)
    if normalized in SERVICES_FIRMS:
        return True

    # Word boundary check for any token matches (handles "TCS Digital" or "Infosys BPM")
    tokens = re.findall(r'\b\w+\b', raw)
    for token in tokens:
        if token in SERVICES_FIRMS:
            return True

    # Check if any known firm name is a substring of the company name
    for firm in SERVICES_FIRMS:
        if len(firm) > 4 and firm in raw:  # min 4 chars to avoid short false matches
            return True

    return False


class ProductVsServicesFeature(BaseFeature):
    """
    Product vs Services score: measures what fraction of a candidate's
    career was spent at product companies vs IT services firms.

    Range: 0.0 (pure services) to 1.0 (pure product)
    reliability: 'clean' if career_history has >= 2 roles, 'sparse' otherwise

    v2 changes:
      - Expanded SERVICES_FIRMS list (9 → ~50 firms)
      - Added normalized matching to handle suffix variations
      - Added substring matching for firm subsidiaries
    """

    def __init__(self):
        super().__init__("product_vs_services", 2, "sparse")  # version bumped to 2

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        career = candidate.get("career_history", [])
        if not career:
            return 0.50, "sparse"  # neutral, no data

        services_months = 0.0
        product_months = 0.0

        for role in career:
            company = role.get("company", "") or ""
            duration = float(role.get("duration_months", 0) or 0)
            if duration <= 0:
                continue

            if _is_services_firm(company):
                services_months += duration
            else:
                product_months += duration

        total = services_months + product_months
        if total == 0:
            return 0.50, "sparse"

        tag = "clean" if len(career) >= 2 else "sparse"
        score = product_months / total
        return score, tag


registry.register(ProductVsServicesFeature())
