import os
import sys
import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES, get_monotonic_constraints

class GBMLambdaRankModel:
    def __init__(self, random_state: int = 42):
        self.features = TRAINING_FEATURES
        self.constraints = get_monotonic_constraints()
        self.model = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=100,
            learning_rate=0.05,
            monotone_constraints=self.constraints,
            random_state=random_state,
            n_jobs=-1
        )

    def extract_features(self, candidates_list: list[dict]) -> np.ndarray:
        """
        Extracts the required features from a list of candidate dictionaries into a numpy array.
        """
        X = []
        for cand in candidates_list:
            row = []
            for f in self.features:
                row.append(float(cand.get(f, 0.0) or 0.0))
            X.append(row)
        return np.array(X)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Fits the LambdaRank model. Since we have a single query (the JD), 
        the group parameter is just the number of training samples.
        """
        n_samples = len(X_train)
        group = []
        while n_samples > 0:
            sz = min(n_samples, 5000)
            group.append(sz)
            n_samples -= sz
            
        self.model.fit(X_train, y_train, group=group, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str):
        """
        Saves the underlying LightGBM booster so it can be loaded directly by SHAP's TreeExplainer.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.booster_.save_model(filepath)

    def load(self, filepath: str):
        """
        Loads the booster. Note that this replaces the LGBMRanker wrapper with a raw booster.
        """
        self.model = lgb.Booster(model_file=filepath)
