from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class DenseIndex:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = 512
        self.corpus_ids = []
        self.embeddings = None

    def _extract_text(self, candidate: Dict[str, Any]) -> str:
        """
        Extract text for Dense Index: summary + headline + skills
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
            
        text = " ".join(parts) if parts else ""
        return text[:1000]

    def build(self, candidates: List[Dict[str, Any]], batch_size: int = 256):
        texts = []
        for cand in candidates:
            self.corpus_ids.append(cand["candidate_id"])
            texts.append(self._extract_text(cand))
            
        print(f"Building dense index over {len(self.corpus_ids)} candidates using {self.model.model_card_data.model_name or 'model'}...")
        
        # We use normalize_embeddings=True for cosine similarity via dot product
        self.embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        print("Dense index built.")

    def search(self, query: str, top_k: int = 200) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("Index not built")
            
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        
        # Dot product for cosine similarity (since embeddings are normalized)
        scores = np.dot(self.embeddings, query_embedding)
        
        # Pair up IDs with scores
        scored_candidates = [(self.corpus_ids[i], float(scores[i])) for i in range(len(self.corpus_ids))]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]
            
        results = []
        for cand_id, score in scored_candidates:
            results.append({"candidate_id": cand_id, "score": score})
            
        return results

    def save(self, path: str):
        print(f"Saving Dense index to {path}...")
        np.savez(path, corpus_ids=np.array(self.corpus_ids), embeddings=self.embeddings)
        
    def load(self, path: str):
        print(f"Loading Dense index from {path}...")
        data = np.load(path)
        self.corpus_ids = data['corpus_ids'].tolist()
        self.embeddings = data['embeddings']
