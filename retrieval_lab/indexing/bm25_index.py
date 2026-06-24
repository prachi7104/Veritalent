from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import pickle
import os

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.corpus_ids = []

    def _extract_text(self, candidate: Dict[str, Any]) -> str:
        """
        Extract text for BM25: skills[].name + current_title + career_history[].title
        """
        parts = []
        
        # skills
        for skill in candidate.get("skills", []):
            if "name" in skill:
                parts.append(skill["name"])
                
        # current_title
        title = candidate.get("profile", {}).get("current_title")
        if title:
            parts.append(title)
            
        # career history titles
        for role in candidate.get("career_history", []):
            role_title = role.get("title")
            if role_title:
                parts.append(role_title)
                
        return " ".join(parts).lower()

    def build(self, candidates: List[Dict[str, Any]]):
        tokenized_corpus = []
        for cand in candidates:
            self.corpus_ids.append(cand["candidate_id"])
            text = self._extract_text(cand)
            tokenized_corpus.append(text.split())
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Built BM25 index over {len(self.corpus_ids)} candidates.")

    def search(self, query: str, top_k: int = 200) -> List[Dict[str, Any]]:
        if not self.bm25:
            raise ValueError("Index not built")
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair up IDs with scores and sort
        scored_candidates = [(self.corpus_ids[i], float(scores[i])) for i in range(len(self.corpus_ids))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for cand_id, score in scored_candidates[:top_k]:
            results.append({"candidate_id": cand_id, "score": score})
            
        return results

    def save(self, path: str):
        print(f"Saving BM25 index to {path}...")
        with open(path, 'wb') as f:
            pickle.dump((self.corpus_ids, self.bm25), f)
            
    def load(self, path: str):
        print(f"Loading BM25 index from {path}...")
        with open(path, 'rb') as f:
            self.corpus_ids, self.bm25 = pickle.load(f)
