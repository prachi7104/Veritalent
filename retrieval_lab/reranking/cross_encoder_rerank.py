from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import time

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def _extract_text(self, candidate: Dict[str, Any]) -> str:
        """
        Extract descriptive text for reranking
        """
        parts = []
        
        title = candidate.get("profile", {}).get("current_title")
        if title:
            parts.append(title)
            
        headline = candidate.get("profile", {}).get("headline")
        if headline:
            parts.append(headline)
            
        summary = candidate.get("profile", {}).get("summary")
        if summary:
            parts.append(summary)
            
        skills = []
        for skill in candidate.get("skills", []):
            if "name" in skill:
                skills.append(skill["name"])
                
        if skills:
            parts.append("Skills: " + ", ".join(skills))
            
        return " | ".join(parts)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], candidates_data: Dict[str, Dict[str, Any]], top_k: int = 200) -> Tuple[List[Dict[str, Any]], float]:
        """
        candidates: List of {"candidate_id": id, "score": initial_score}
        candidates_data: mapping of candidate_id to full candidate dict
        Returns (reranked_results, latency_ms)
        """
        if not candidates:
            return [], 0.0
            
        to_rerank = candidates[:top_k]
        
        pairs = []
        for cand in to_rerank:
            cand_dict = candidates_data[cand["candidate_id"]]
            text = self._extract_text(cand_dict)
            pairs.append([query, text])
            
        start_time = time.time()
        scores = self.model.predict(pairs)
        latency_ms = (time.time() - start_time) * 1000
        
        reranked = []
        for i, cand in enumerate(to_rerank):
            reranked.append({
                "candidate_id": cand["candidate_id"],
                "score": float(scores[i])
            })
            
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked, latency_ms
