from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class MultiVectorIndex:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.corpus_ids = []
        self.embeddings = {
            "summary": None,
            "headline": None,
            "skills": None
        }

    def _extract_texts(self, candidate: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract separate texts for Multi-Vector Index
        """
        summary = candidate.get("profile", {}).get("summary", "")
        headline = candidate.get("profile", {}).get("headline", "")
        
        skills = []
        for skill in candidate.get("skills", []):
            if "name" in skill:
                skills.append(skill["name"])
        skills_text = ", ".join(skills)
        
        return {
            "summary": summary,
            "headline": headline,
            "skills": skills_text
        }

    def build(self, candidates: List[Dict[str, Any]], batch_size: int = 256):
        summaries = []
        headlines = []
        skills_lists = []
        
        for cand in candidates:
            self.corpus_ids.append(cand["candidate_id"])
            texts = self._extract_texts(cand)
            summaries.append(texts["summary"])
            headlines.append(texts["headline"])
            skills_lists.append(texts["skills"])
            
        print(f"Building multi-vector index over {len(self.corpus_ids)} candidates...")
        
        self.embeddings["summary"] = self.model.encode(summaries, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings["headline"] = self.model.encode(headlines, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings["skills"] = self.model.encode(skills_lists, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        print("Multi-vector index built.")

    def search(self, jd_queries: Dict[str, str], weights: Dict[str, float] = None, top_k: int = 200) -> List[Dict[str, Any]]:
        """
        jd_queries: dict like {"summary": "narrative...", "skills": "required skills..."}
        """
        if self.embeddings["summary"] is None:
            raise ValueError("Index not built")
            
        if weights is None:
            weights = {"summary": 1.0, "headline": 1.0, "skills": 1.0}
            
        total_scores = np.zeros(len(self.corpus_ids))
        
        for field, query_text in jd_queries.items():
            if field in self.embeddings and query_text:
                query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0]
                field_scores = np.dot(self.embeddings[field], query_embedding)
                total_scores += field_scores * weights.get(field, 1.0)
                
        # Pair up IDs with scores
        scored_candidates = [(self.corpus_ids[i], float(total_scores[i])) for i in range(len(self.corpus_ids))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for cand_id, score in scored_candidates[:top_k]:
            results.append({"candidate_id": cand_id, "score": score})
            
        return results

    def save(self, path: str):
        print(f"Saving MultiVector index to {path}...")
        np.savez(path, 
                 corpus_ids=np.array(self.corpus_ids), 
                 summary_embeddings=self.embeddings["summary"],
                 headline_embeddings=self.embeddings["headline"],
                 skills_embeddings=self.embeddings["skills"])
        
    def load(self, path: str):
        print(f"Loading MultiVector index from {path}...")
        data = np.load(path)
        self.corpus_ids = data['corpus_ids'].tolist()
        self.embeddings["summary"] = data['summary_embeddings']
        self.embeddings["headline"] = data['headline_embeddings']
        self.embeddings["skills"] = data['skills_embeddings']
