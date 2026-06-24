from typing import Any, Dict, Tuple
from .base import BaseFeature
from .feature_registry import registry

IMPLIED_SKILL_PHRASES = {
    "ranking_systems": [
        "ranking pipeline", "ranking system", "learning to rank", "ltr",
        "search ranking", "reranking", "result ranking", "rank optimization"
    ],
    "information_retrieval": [
        "information retrieval", "search infrastructure", "search backend",
        "retrieval system", "search engine", "document retrieval",
        "information extraction"
    ],
    "vector_search": [
        "vector search", "semantic search", "embedding search",
        "dense retrieval", "approximate nearest neighbor", "ann index",
        "similarity search"
    ],
    "search_engineering": [
        "search platform", "search quality", "search relevance",
        "query understanding", "indexing pipeline", "inverted index",
        "search infrastructure"
    ],
    "production_ml": [
        "shipped to production", "production ml", "ml in production",
        "deployed model", "model serving", "real-time inference",
        "ml pipeline", "feature pipeline", "online learning"
    ]
}

class ImpliedSkillFeatures(BaseFeature):
    """
    Scans candidate summary and headline to detect implied deep-IR skills
    that aren't explicitly listed in the skills array.
    """
    
    def __init__(self):
        super().__init__(
            name="implied_skill_score",
            version=1,
            default_reliability_tag="clean"
        )
        
    def compute(self, candidate: Dict[str, Any]) -> Tuple[Any, str]:
        profile = candidate.get("profile", {})
        summary = profile.get("summary", "") or ""
        headline = profile.get("headline", "") or ""
        
        combined_text = f"{summary} {headline}".lower()
        
        matched_categories = []
        
        for category, phrases in IMPLIED_SKILL_PHRASES.items():
            for phrase in phrases:
                if phrase.lower() in combined_text:
                    matched_categories.append(category)
                    break # Match once per category
                    
        total_categories = len(IMPLIED_SKILL_PHRASES)
        score = len(matched_categories) / total_categories if total_categories > 0 else 0.0
        
        # Pipe-delimited to survive JSON serialization easily
        categories_str = "|".join(matched_categories)
        
        output = {
            "implied_skill_score": float(score),
            "implied_skill_categories": categories_str
        }
        
        return output, "clean"

registry.register(ImpliedSkillFeatures())