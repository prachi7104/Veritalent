from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time

class ShootoutIndex:
    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = SentenceTransformer(model_name)
        self.corpus_ids = []
        self.embeddings = None
        self.indexing_time_s = 0.0

    def _extract_text(self, candidate: Dict[str, Any]) -> str:
        """
        Extract text for Shootout Index: summary + headline + skills (if any)
        We include skills here because the shootout specifies text fields (summary, headline, skills).
        """
        parts = []
        
        # summary
        summary = candidate.get("profile", {}).get("summary")
        if summary:
            parts.append(summary)
            
        # headline
        headline = candidate.get("profile", {}).get("headline")
        if headline:
            parts.append(headline)
            
        # skills
        skills = [s['name'] for s in candidate.get('skills', []) if isinstance(s, dict) and 'name' in s]
        if skills:
            parts.append("Skills: " + ", ".join(skills))
            
        full_text = " ".join(parts)
        # TRUNCATE to 1000 characters to prevent O(N^2) attention from destroying CPU inference time
        text = full_text[:1000] if parts else ""
        return self.passage_prefix + text

    def build(self, candidates: List[Dict[str, Any]], batch_size: int = 256):
        import time
        start_time = time.time()
        
        self.corpus_ids = []
        texts = []
        
        for cand in candidates:
            self.corpus_ids.append(cand["candidate_id"])
            texts.append(self._extract_text(cand))
            
        print(f"Building dense index over {len(texts)} candidates using {self.model.model_card_data.model_name}...")
        
        self.embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        
        self.indexing_time_s = time.time() - start_time
        print(f"Index built in {self.indexing_time_s:.2f}s.")

    def search(self, query: str, top_k: int = 200) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("Index not built")
            
        prefixed_query = self.query_prefix + query
        query_embedding = self.model.encode([prefixed_query], normalize_embeddings=True)[0]
        
        # Dot product for cosine similarity (since embeddings are normalized)
        scores = np.dot(self.embeddings, query_embedding)
        
        # Pair up IDs with scores
        scored_candidates = [(self.corpus_ids[i], float(scores[i])) for i in range(len(self.corpus_ids))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for cand_id, score in scored_candidates[:top_k]:
            results.append({"candidate_id": cand_id, "score": score})
            
        return results

    def save(self, path: str):
        print(f"Saving Shootout index to {path}...")
        np.savez(path, corpus_ids=np.array(self.corpus_ids), embeddings=self.embeddings, indexing_time_s=np.array([self.indexing_time_s]))
        
    def load(self, path: str):
        print(f"Loading Shootout index from {path}...")
        data = np.load(path)
        self.corpus_ids = data['corpus_ids'].tolist()
        self.embeddings = data['embeddings']
        if 'indexing_time_s' in data:
            self.indexing_time_s = data['indexing_time_s'][0]
