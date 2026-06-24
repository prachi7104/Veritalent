import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LearnedFusion:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, gold_set_path: str, bm25_scores: Dict[str, float], dense_scores: Dict[str, float]):
        """
        Trains the fusion model on raw scores against the gold set labels.
        gold_set_path: path to gold_set.json
        bm25_scores: dict mapping candidate_id to bm25 score
        dense_scores: dict mapping candidate_id to dense score
        """
        with open(gold_set_path, 'r') as f:
            gold_data = json.load(f)
            
        judgments = gold_data["queries"][0]["judgments"]
        
        X = []
        y = []
        
        for cand_id, score in judgments.items():
            bm25 = bm25_scores.get(cand_id, 0.0)
            dense = dense_scores.get(cand_id, 0.0)
            X.append([bm25, dense])
            # Binarize label for logistic regression: >= 2 is relevant (1), else 0
            y.append(1 if score >= 2 else 0)
            
        if sum(y) == 0 or sum(y) == len(y):
            print("Warning: Only one class present in training data for learned fusion.")
            # We still fit to avoid errors, but it won't be meaningful
            
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(f"Learned Fusion model trained. Coefficients: {self.model.coef_[0]}")

    def score(self, bm25_scores: Dict[str, float], dense_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("Learned fusion model not trained.")
            
        all_cands = set(bm25_scores.keys()).union(set(dense_scores.keys()))
        X = []
        cand_ids = []
        
        for cand_id in all_cands:
            bm25 = bm25_scores.get(cand_id, 0.0)
            dense = dense_scores.get(cand_id, 0.0)
            X.append([bm25, dense])
            cand_ids.append(cand_id)
            
        if not X:
            return []
            
        X_scaled = self.scaler.transform(X)
        
        # Get probability of class 1 (relevant)
        try:
            probas = self.model.predict_proba(X_scaled)[:, 1]
        except IndexError:
            # If the model only learned one class
            probas = self.model.predict(X_scaled)
            
        results = []
        for i, cand_id in enumerate(cand_ids):
            results.append({"candidate_id": cand_id, "score": float(probas[i])})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
