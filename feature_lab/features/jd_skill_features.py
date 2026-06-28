"""
jd_skill_features.py — JD-aligned skill scoring.

Replaces skill_mastery_triangulation as primary skill signal.
Key difference: weights skills by their relevance to THIS JD's requirements,
not just by duration × proficiency.

Band weights (JD-calibrated for Senior AI/ML Engineer, Search & Retrieval):
  deep-ir:     3.0  — PyTorch, BM25, Elasticsearch, LtR, Qdrant, etc.
               These are the JD's literal must-haves.
  buzzword:    1.0  — RAG, LangChain, FAISS, Embeddings
               Necessary but not sufficient; JD treats these as table stakes.
  fingerprint: 2.0  — Search Backend, Ranking Systems, Text Encoders
               Ultra-rare, direct match to JD specialty.
  generic:     0.1  — HTML, Excel, etc.
               Noise for this JD.

Duration cap: 60 months (5 years). Beyond this, additional time doesn't
add proportional value — seniority is already captured by YOE feature.

Formula per skill:
  base_score = min(duration_months, 60) × proficiency_weight × band_weight
  if assessment_score available: blend 60% heuristic + 40% assessed
  add endorsements × 0.3 × band_weight (small bonus, band-scaled)

Final score: sum across ALL relevant skills (not max).
  Rationale: a candidate with breadth across deep-IR tools is more valuable
  than one with extreme depth in a single tool.
"""
from typing import Dict, Any, Tuple

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry
from feature_lab.features.skill_features import (
    get_skill_band, PROFICIENCY_WEIGHTS
)

DURATION_CAP_MONTHS = 60  # 5 years cap

JD_BAND_WEIGHTS = {
    "deep-ir":     3.0,
    "fingerprint": 2.0,
    "buzzword":    1.0,
    "generic":     0.1,
}


class JDWeightedSkillScoreFeature(BaseFeature):
    """
    JD-aligned skill score: sum of (duration × proficiency × band_weight)
    across all skills, with duration capped at 60 months.

    Unlike skill_mastery_triangulation (which takes MAX across skills),
    this SUMS across all skills to reward genuine breadth within the
    JD-relevant skill space.

    reliability:
      'clean'  — when at least one deep-IR skill has assessment corroboration
      'sparse' — otherwise
    """

    def __init__(self):
        super().__init__("jd_skill_score", 1, "sparse")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        skills = candidate.get("skills", [])
        assessments = (
            candidate.get("redrob_signals", {})
            .get("skill_assessment_scores", {}) or {}
        )

        total_score = 0.0
        has_clean_assessment = False

        for skill in skills:
            name = skill.get("name", "")
            if not name:
                continue

            band = get_skill_band(name)
            band_weight = JD_BAND_WEIGHTS.get(band, 0.1)

            # Skip generic skills entirely (0.1 weight is tiny but not zero)
            # Only skip if band is not in our target set
            prof = skill.get("proficiency", "intermediate").lower()
            prof_weight = PROFICIENCY_WEIGHTS.get(prof, 1.0)

            dur = float(skill.get("duration_months", 0) or 0)
            dur_capped = min(dur, DURATION_CAP_MONTHS)

            # Base heuristic
            base = dur_capped * prof_weight * band_weight

            # Endorsement bonus (band-scaled, small)
            endorsements = float(skill.get("endorsements", 0) or 0)
            endorse_bonus = endorsements * 0.3 * band_weight

            skill_score = base + endorse_bonus

            # Assessment corroboration (for deep-IR skills only)
            if name in assessments and band in ("deep-ir", "fingerprint"):
                ass_score = float(assessments[name])
                # Blend: 60% heuristic, 40% assessed
                skill_score = skill_score * 0.6 + ass_score * 0.4 * band_weight
                has_clean_assessment = True

            total_score += skill_score

        tag = "clean" if has_clean_assessment else "sparse"
        return total_score, tag


registry.register(JDWeightedSkillScoreFeature())
