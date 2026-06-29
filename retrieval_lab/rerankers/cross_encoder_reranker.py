"""
Cross-Encoder Reranker.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, 6 transformer layers
  - ~50ms per candidate on CPU
  - Trained on MS MARCO passage relevance

Why MiniLM-L6:
  - 4x faster than BERT-base
  - ~95% of BERT-base quality on reranking
  - 50 candidates in ~2.5s on CPU — within 2s latency budget at top-20 scope

Important limitation:
  This model scores (JD, candidate_text) semantic relevance ONLY.
  It cannot see notice_period, location, or trust_score.
  Always combine with GBM scores for final ranking.
"""
import time
from typing import List, Tuple

import numpy as np

CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_candidate_text(candidate: dict) -> str:
    """
    Compact candidate text for the cross-encoder input.
    Priority: headline > summary > recent roles > JD-relevant skills > YOE.
    Caps at ~400 tokens to stay within 512-token limit.
    """
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])

    parts = []

    headline = profile.get("headline", "")
    if headline:
        parts.append(f"Title: {headline}")

    summary = profile.get("summary", "")
    if summary:
        # 350 char cap — takes ~90 tokens, leaves room for skills
        parts.append(f"Summary: {summary[:350]}")

    for role in career[:2]:
        t = role.get("title", "")
        c = role.get("company", "")
        d = role.get("duration_months", "")
        if t or c:
            parts.append(f"Role: {t} at {c} ({d}m)")

    try:
        from feature_lab.features.skill_features import get_skill_band
        deep  = [s["name"] for s in skills
                 if get_skill_band(s.get("name","")) in ("deep-ir","fingerprint")]
        other = [s["name"] for s in skills
                 if get_skill_band(s.get("name","")) not in ("deep-ir","fingerprint")]
        skill_str = ", ".join(deep[:8] + other[:3])
    except ImportError:
        skill_str = ", ".join(s.get("name","") for s in skills[:8])

    if skill_str:
        parts.append(f"Skills: {skill_str}")

    yoe = profile.get("years_of_experience")
    if yoe:
        parts.append(f"YOE: {yoe}")

    return " | ".join(parts)


class CrossEncoderReranker:
    """
    Re-ranks candidates by joint (JD, candidate) relevance.
    Returns reranked IDs, scores, and latency_ms.
    """

    def __init__(self, model_name: str = CE_MODEL_NAME):
        try:
            from sentence_transformers import CrossEncoder as CE
            self.model = CE(model_name, max_length=512)
            self._available = True
        except ImportError:
            self._available = False
            print("WARNING: sentence-transformers not installed. CE reranker disabled.")

    def rerank(
        self,
        jd_text: str,
        candidates: List[dict],
        candidate_ids: List[str],
        top_n: int = 20,
    ) -> Tuple[List[str], List[float], float]:
        """
        Args:
            jd_text: Raw JD text
            candidates: Candidate profile dicts (same order as candidate_ids)
            candidate_ids: Candidate IDs
            top_n: Number to rerank (only first top_n are reranked; rest unchanged)

        Returns: (reranked_ids_full, reranked_scores_top_n, latency_ms)
        """
        assert len(candidates) == len(candidate_ids)
        if not self._available:
            return candidate_ids, [0.0] * len(candidate_ids), 0.0

        to_rerank = candidates[:top_n]
        to_rerank_ids = candidate_ids[:top_n]
        rest_ids = candidate_ids[top_n:]

        t0 = time.time()
        pairs  = [[jd_text, build_candidate_text(c)] for c in to_rerank]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order  = np.argsort(-np.array(scores))

        reranked_ids    = [to_rerank_ids[i] for i in order]
        reranked_scores = [float(scores[i])  for i in order]
        latency_ms = (time.time() - t0) * 1000

        return reranked_ids + rest_ids, reranked_scores, latency_ms
