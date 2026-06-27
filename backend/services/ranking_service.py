import os
import logging
import numpy as np
import lightgbm as lgb
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES
from ranking_lab.models.linear_baseline import LinearBaselineModel
from backend.services.scenario_service import scenario_rerank

logger = logging.getLogger(__name__)

MODEL_PATH = "ranking_lab/models/gbm_lambdarank.txt"
_booster: lgb.Booster | None = None
_linear_fallback = LinearBaselineModel()

def load():
    global _booster
    logger.info(f"Loading GBM model from {MODEL_PATH}...")
    try:
        if os.path.exists(MODEL_PATH):
            _booster = lgb.Booster(model_file=MODEL_PATH)
            logger.info("GBM booster loaded.")
        else:
            logger.error(f"GBM model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"GBM model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load GBM booster: {e}")
        raise

def is_loaded() -> bool:
    return _booster is not None

def score_batch(feature_dicts: list[dict]) -> list[float]:
    if not feature_dicts:
        return []

    X = []
    for candidate_feat in feature_dicts:
        row = []
        for feature_name in TRAINING_FEATURES:
            row.append(float(candidate_feat.get(feature_name, 0.0)))
        
        assert len(row) == len(TRAINING_FEATURES), f"Feature count mismatch: {len(row)} vs {len(TRAINING_FEATURES)}"
        X.append(row)
        
    X_array = np.array(X)
    
    try:
        if not _booster:
            raise RuntimeError("GBM booster not loaded")
        scores = _booster.predict(X_array)
        return scores.tolist()
    except Exception as e:
        logger.warning(f"GBM prediction failed: {e}. Falling back to LinearBaselineModel.")
        return _score_batch_linear_fallback(feature_dicts)

def _score_batch_linear_fallback(feature_dicts: list[dict]) -> list[float]:
    scores_dict = _linear_fallback.predict_from_dicts(feature_dicts)
    return [scores_dict.get(fd.get("candidate_id"), 0.0) for fd in feature_dicts]

def linear_score(session_id: str, weight_overrides: dict) -> list[dict]:
    return scenario_rerank(session_id, weight_overrides)
