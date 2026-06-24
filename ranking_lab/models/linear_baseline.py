import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ranking_lab.labels.synthetic_formula_labels import normalize_features, FEATURE_GROUPS, GROUP_WEIGHTS

class LinearBaselineModel:
    def __init__(self):
        self.feature_groups = FEATURE_GROUPS
        self.group_weights = GROUP_WEIGHTS

    def predict_from_dicts(self, candidates_list: list[dict]) -> dict[str, float]:
        """
        Returns a dictionary mapping candidate_id to their linear baseline score.
        """
        # Create a deep copy mapped by ID for the normalizer
        cands_dict = {c["candidate_id"]: dict(c) for c in candidates_list}
        normalize_features(cands_dict)
        
        scores = {}
        for cid, feats in cands_dict.items():
            score = 0.0
            for group, group_feats in self.feature_groups.items():
                g_sum = sum(feats.get(f"{f}_norm", 0.0) for f in group_feats)
                score += g_sum * self.group_weights.get(group, 0.0)
            scores[cid] = score
            
        return scores
