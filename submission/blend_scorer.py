# submission/blend_scorer.py
"""
BlendScorer v2 — simplified formula after ablation.
final_score = alpha × norm(gbm_score) + (1-alpha) × norm(jd_skill_score)

yoe_band_fit removed: ablation proved zero marginal contribution.
jd_skill_score takes full (1-alpha) = 10% slice.
"""
import json
import numpy as np
from pathlib import Path


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx == mn else (arr - mn) / (mx - mn)


class BlendScorer:
    def __init__(self,
                 config_path: str = "ranking_lab/models/blend_config.json",
                 model_path:  str = "ranking_lab/models/gbm_lambdarank.txt"):
        with open(config_path) as f:
            cfg = json.load(f)
        self.alpha     = float(cfg["alpha"])
        self.jd_weight = float(cfg["new_features"]["jd_skill_score"])
        # yoe_weight intentionally not read — ablation proved it's 0

        from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
        from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES
        self.model = GBMLambdaRankModel()
        self.model.load(model_path)
        num_feats = self.model.model.num_feature()
        self.training_features = TRAINING_FEATURES[:num_feats]

        print(f"[BlendScorer] alpha={self.alpha}, jd_weight={self.jd_weight}")
        print(f"[BlendScorer] Formula: {self.alpha}×GBM + {1-self.alpha}×jd_skill")

    def score(self, candidate_ids: list, feature_store: dict) -> dict:
        X_rows, valid_ids = [], []
        for cid in candidate_ids:
            if cid not in feature_store:
                continue
            feats = feature_store[cid]
            X_rows.append([float(feats.get(f, 0.0) or 0.0)
                           for f in self.training_features])
            valid_ids.append(cid)

        if not X_rows:
            return {}

        X = np.array(X_rows)
        gbm_norm = _norm(self.model.predict(X))

        jd_raw  = np.array([float(feature_store[c].get("jd_skill_score", 0) or 0)
                             for c in valid_ids])
        jd_norm = _norm(jd_raw)

        blend = self.alpha * gbm_norm + (1 - self.alpha) * jd_norm
        return {cid: float(blend[i]) for i, cid in enumerate(valid_ids)}
