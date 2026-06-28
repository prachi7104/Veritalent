"""
skill_features.py — Skill depth, breadth, recency, and mastery features.

Skill Band Definitions (from master context, Section 2.5):
  - generic: ~12,000 occurrences each — HTML, Excel, Salesforce CRM, etc. Noise for this JD.
  - buzzword: ~4,700–5,200 occurrences — RAG, LangChain, FAISS, Embeddings, etc.
              Necessary but not sufficient (keyword-stuffer red flag if only these).
  - deep-ir: ~1,300–1,400 occurrences — PyTorch, BM25, Elasticsearch, LoRA, etc.
             Maps 1:1 to JD's actual must-haves.
  - fingerprint: <7 occurrences — ultra-rare tail, treated as capped tiebreaker only.

Note on skill_recency:
  Approximates "currently using" by checking whether a skill's duration_months
  extends into the current role's tenure window, using cumulative duration
  of past roles as a proxy for the start of the current role.
"""
from typing import Dict, Any, Tuple
from datetime import date, datetime

from feature_lab.features.base import BaseFeature
from feature_lab.features.feature_registry import registry

# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------
DEEP_IR_SKILLS = {
    "PyTorch", "TensorFlow", "NLP", "Machine Learning", "Deep Learning",
    "BM25", "Learning to Rank", "Qdrant", "Weaviate", "Milvus",
    "scikit-learn", "Elasticsearch", "OpenSearch", "LlamaIndex",
    "Haystack", "QLoRA", "PEFT", "LoRA", "pgvector", "Natural Language Processing",
}

FINGERPRINT_SKILLS = {
    "Search Backend", "Ranking Systems", "Text Encoders", "Vector Representations",
    "Content Matching", "Model Adaptation", "Information Retrieval Systems",
    "Search & Discovery", "Search Infrastructure", "Indexing Algorithms",
    "Workflow Orchestration", "Open-source ML libraries", "Document Processing",
}

BUZZWORD_SKILLS = {
    "RAG", "LangChain", "Pinecone", "FAISS", "Embeddings", "Computer Vision",
    "LLMs", "Generative AI", "Prompt Engineering", "ChatGPT",
}

PROFICIENCY_WEIGHTS = {
    "beginner": 0.5,
    "intermediate": 1.0,
    "advanced": 1.5,
    "expert": 2.0,
}


def get_skill_band(name: str) -> str:
    """Return the frequency-band label for a skill name."""
    n = name.strip()
    if n in DEEP_IR_SKILLS:
        return "deep-ir"
    if n in FINGERPRINT_SKILLS:
        return "fingerprint"
    if n in BUZZWORD_SKILLS:
        return "buzzword"
    return "generic"


# ---------------------------------------------------------------------------
# Feature implementations
# ---------------------------------------------------------------------------

class SkillDepthFeature(BaseFeature):
    """
    Max duration_months × proficiency_weight among deep-IR band skills.
    Proxy for "how long have they actually worked with the relevant stack".
    reliability: clean (based on stated duration, no assessment needed)
    """
    def __init__(self):
        super().__init__("skill_depth", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        skills = candidate.get("skills", [])
        max_depth = 0.0
        for skill in skills:
            if get_skill_band(skill.get("name", "")) == "deep-ir":
                prof = skill.get("proficiency", "intermediate").lower()
                weight = PROFICIENCY_WEIGHTS.get(prof, 1.0)
                dur = float(skill.get("duration_months", 0) or 0)
                score = dur * weight
                if score > max_depth:
                    max_depth = score
        return max_depth, self.default_reliability_tag


class SkillBreadthFeature(BaseFeature):
    """
    Fraction of skills that span ≥2 frequency bands.
    Penalises pure keyword-stuffers (all buzzwords, no deep-IR).
    reliability: clean
    """
    def __init__(self):
        super().__init__("skill_breadth", 1, "clean")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        skills = candidate.get("skills", [])
        if not skills:
            return 0.0, self.default_reliability_tag

        distinct_bands: set = set()
        for skill in skills:
            distinct_bands.add(get_skill_band(skill.get("name", "")))

        # Score = proportion of bands covered (max 4 bands)
        breadth = len(distinct_bands) / 4.0
        return breadth, self.default_reliability_tag


class SkillRecencyFeature(BaseFeature):
    """
    Skill recency: fraction of deep-IR and buzzword skills that appear
    to be currently in use, based on career timeline.

    NEW LOGIC (v2):
      A skill is considered "recent" if its stated duration_months extends
      into the last 24 months of the candidate's career timeline.

      Method:
        1. Compute candidate's career_start estimate:
           today_months - years_of_experience * 12
        2. A skill is recent if:
           career_start + skill.duration_months > today_months - 24
           i.e. the skill's duration would reach into the last 2 years

      Returns: fraction of deep-IR + buzzword skills that are recent.
      Range: 0.0 to 1.0
      reliability: clean

    WHY: Old binary logic (duration > cumulative_past) failed for all
    senior candidates with long career histories. This date-anchored
    approach correctly handles experienced engineers.
    """

    def __init__(self):
        super().__init__("skill_recency", 2, "clean")  # version bumped to 2

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        today_months = date.today().year * 12 + date.today().month

        # Estimate career start from YOE
        yoe = float(candidate.get("profile", {}).get("years_of_experience", 0) or 0)
        if yoe <= 0:
            return 0.0, self.default_reliability_tag

        career_start_months = today_months - int(yoe * 12)
        recency_cutoff = today_months - 24  # 2 years ago

        skills = candidate.get("skills", [])
        target_bands = ("deep-ir", "buzzword")
        target_skills = [s for s in skills
                         if get_skill_band(s.get("name", "")) in target_bands]

        if not target_skills:
            return 0.0, self.default_reliability_tag

        recent_count = 0
        for skill in target_skills:
            dur = int(skill.get("duration_months", 0) or 0)
            if dur <= 0:
                continue
            # skill's effective end month in career timeline
            skill_end_month = career_start_months + dur
            if skill_end_month >= recency_cutoff:
                recent_count += 1

        return recent_count / len(target_skills), self.default_reliability_tag


class SkillMasteryTriangulationFeature(BaseFeature):
    """
    Per-candidate max mastery estimate across all skills, combining:
      - proficiency label (weight 0.5×)
      - duration_months (scaled)
      - endorsements (bonus)
      - skill_assessment_score (corroboration, when available)

    reliability_tag:
      'clean'  — when an assessment score corroborates the best skill
      'sparse' — otherwise (no assessment coverage)
    """
    def __init__(self):
        super().__init__("skill_mastery_triangulation", 1, "sparse")

    def compute(self, candidate: Dict[str, Any]) -> Tuple[float, str]:
        skills = candidate.get("skills", [])
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}

        max_mastery = 0.0
        best_tag = "sparse"

        for skill in skills:
            name = skill.get("name", "")
            if not name:
                continue

            prof = skill.get("proficiency", "intermediate").lower()
            weight = PROFICIENCY_WEIGHTS.get(prof, 1.0)
            dur = float(skill.get("duration_months", 0) or 0)
            endorsements = float(skill.get("endorsements", 0) or 0)

            # Base mastery: duration × proficiency + endorsements bonus
            mastery = (dur * weight) + (endorsements * 0.5)

            tag = "sparse"
            if name in assessments:
                ass_score = float(assessments[name])
                # Blend heuristic (50%) with assessed score (50%)
                mastery = (mastery * 0.5) + (ass_score * 0.5)
                tag = "clean"

            if mastery > max_mastery:
                max_mastery = mastery
                best_tag = tag

        return max_mastery, best_tag


# ---------------------------------------------------------------------------
# Register all features
# ---------------------------------------------------------------------------
registry.register(SkillDepthFeature())
registry.register(SkillBreadthFeature())
registry.register(SkillRecencyFeature())
registry.register(SkillMasteryTriangulationFeature())
