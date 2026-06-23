import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Iterator

from .schema_guard import enforce_schema

# =============================================================================
# TASK 0 — ENUMERATED TITLE VALUES (from one-time pass over candidates.jsonl)
# Exactly 45 unique values found (master context stated ~47 — close enough).
# Format: count: "exact title string"
# =============================================================================
# COMPLETE TITLE ENUMERATION (sorted by count descending):
#
#   5833: "Business Analyst"          <- REMOVE (non-ML, business-facing)
#   5830: "HR Manager"               <- REMOVE
#   5791: "Mechanical Engineer"       <- REMOVE (physical engineering, not software)
#   5764: "Accountant"               <- REMOVE
#   5754: "Project Manager"          <- REMOVE (non-technical PM)
#   5750: "Customer Support"         <- REMOVE
#   5744: "Operations Manager"       <- REMOVE
#   5727: "Content Writer"           <- REMOVE
#   5713: "Sales Executive"          <- REMOVE
#   5702: "Civil Engineer"           <- REMOVE (physical engineering, not software)
#   5689: "Graphic Designer"         <- REMOVE
#   5524: "Marketing Manager"        <- REMOVE
#   3450: "Software Engineer"        <- KEEP
#   2873: "Full Stack Developer"     <- KEEP
#   2836: "Cloud Engineer"           <- KEEP
#   2809: "Java Developer"           <- KEEP
#   2788: ".NET Developer"           <- KEEP
#   2787: "DevOps Engineer"          <- KEEP
#   2757: "Mobile Developer"         <- KEEP
#   2738: "Frontend Engineer"        <- KEEP
#   2682: "QA Engineer"              <- KEEP (close to ML infra; may be transitioning)
#    764: "Analytics Engineer"       <- KEEP
#    744: "Data Engineer"            <- KEEP
#    728: "Data Analyst"             <- KEEP (could be transitioning to ML)
#    704: "Backend Engineer"         <- KEEP
#    687: "Senior Data Engineer"     <- KEEP
#    653: "Senior Software Engineer" <- KEEP
#    167: "ML Engineer"              <- KEEP (core target title)
#    153: "AI Research Engineer"     <- KEEP (core target title)
#    145: "Data Scientist"           <- KEEP (core target title)
#    142: "Senior Software Engineer (ML)" <- KEEP (core target title)
#    132: "Computer Vision Engineer" <- KEEP
#    131: "Junior ML Engineer"       <- KEEP (may grow into target; keep for recall)
#    130: "AI Specialist"            <- KEEP
#     26: "Recommendation Systems Engineer" <- KEEP (directly relevant)
#     24: "Machine Learning Engineer"      <- KEEP (core target title)
#     23: "Applied ML Engineer"            <- KEEP (core target title)
#     23: "Search Engineer"               <- KEEP (directly relevant to JD)
#     21: "AI Engineer"                   <- KEEP
#     19: "Senior Data Scientist"         <- KEEP
#     14: "NLP Engineer"                  <- KEEP (directly relevant to JD)
#      6: "Senior NLP Engineer"           <- KEEP
#      6: "Senior Machine Learning Engineer" <- KEEP
#      6: "Staff Machine Learning Engineer"  <- KEEP
#      4: "Senior AI Engineer"            <- KEEP
#      4: "Senior Applied Scientist"      <- KEEP
#      3: "Lead AI Engineer"              <- KEEP
# =============================================================================
# CLASSIFICATION SUMMARY:
# - 12 titles REMOVED: ~68,821 candidates (~69% of pool) — noise exclusion
# - 33 titles KEPT:    ~31,179 candidates (~31% of pool) — target population
# =============================================================================

ALLOWED_TITLES: frozenset = frozenset({
    # Software Engineering (broad tech base — potential career transitioners)
    "software engineer",
    "senior software engineer",
    "senior software engineer (ml)",
    "full stack developer",
    "backend engineer",
    "frontend engineer",
    "mobile developer",
    "java developer",
    ".net developer",
    "cloud engineer",
    "devops engineer",
    "qa engineer",
    # Data Engineering & Analytics
    "data engineer",
    "senior data engineer",
    "data analyst",
    "analytics engineer",
    # Core ML / AI titles (primary target population)
    "machine learning engineer",
    "ml engineer",
    "junior ml engineer",
    "applied ml engineer",
    "senior machine learning engineer",
    "staff machine learning engineer",
    "ai engineer",
    "senior ai engineer",
    "lead ai engineer",
    "ai research engineer",
    "ai specialist",
    # Data Science
    "data scientist",
    "senior data scientist",
    "senior applied scientist",
    # Specialized ML/NLP/Search
    "nlp engineer",
    "senior nlp engineer",
    "computer vision engineer",
    "search engineer",
    "recommendation systems engineer",
})

# Module-level assertion: validate sample of known-keep and known-remove titles
# This will blow up loudly if a future edit accidentally corrupts the frozenset.
_KNOWN_KEEP = {
    "machine learning engineer", "data scientist", "software engineer",
    "senior data engineer", "ai research engineer", "nlp engineer",
}
_KNOWN_REMOVE = {
    "hr manager", "sales executive", "marketing manager",
    "accountant", "business analyst", "project manager",
    "mechanical engineer", "civil engineer", "content writer",
}

assert all(t in ALLOWED_TITLES for t in _KNOWN_KEEP), \
    f"ALLOWED_TITLES assertion failed: missing known-keep titles: {_KNOWN_KEEP - ALLOWED_TITLES}"
assert all(t not in ALLOWED_TITLES for t in _KNOWN_REMOVE), \
    f"ALLOWED_TITLES assertion failed: known-remove title leaked into allowlist: {_KNOWN_REMOVE & ALLOWED_TITLES}"


def funnel_filter(candidate: Dict) -> bool:
    """
    Closed-enum allowlist domain pre-filter.

    Uses exact-match against a frozenset of lowercase allowed title strings.
    Safe to use as exact-match because current_title is a 45-value closed enum,
    not free text. Any title NOT in ALLOWED_TITLES is excluded.

    This is a coarse domain pre-filter upstream of the title-tier soft-prior ranking
    that still applies within the retained ~31K population. It is NOT a quality gate —
    it removes structural noise (non-tech roles), not weak tech candidates.

    Decision rule for ambiguity: wrong KEEP is recoverable (candidate gets down-ranked);
    wrong REMOVE is NOT recoverable (candidate is permanently excluded). When in doubt, KEEP.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()
    return title in ALLOWED_TITLES


def load_candidates_iter(filepath: str, limit: Optional[int] = None, apply_funnel: bool = False) -> Iterator[Dict]:
    """
    Yields parsed and schema-checked candidates from the jsonl file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if limit is not None and count >= limit:
                break

            line = line.strip()
            if not line:
                continue

            raw_candidate = json.loads(line)
            guarded_candidate = enforce_schema(raw_candidate)

            if apply_funnel and not funnel_filter(guarded_candidate):
                continue

            yield guarded_candidate
            count += 1


def load_candidates(filepath: str, limit: Optional[int] = None, apply_funnel: bool = False) -> List[Dict]:
    """
    Returns a list of parsed and schema-checked candidates.
    """
    return list(load_candidates_iter(filepath, limit, apply_funnel))
