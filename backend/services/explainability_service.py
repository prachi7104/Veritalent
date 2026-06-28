import os
import json
import logging
from explainability_lab.attribution.shap_explainer import SHAPExplainer
from explainability_lab.narrative.fallback_narrative import generate_fallback_narrative
from backend.api.schemas.responses import FeatureContribution

logger = logging.getLogger(__name__)

CACHE_DIR = "explainability_lab/narratives_cache"
_explainer = None
_narratives_count = 0

def load():
    global _explainer, _narratives_count
    logger.info("Loading SHAP explainer...")
    try:
        _explainer = SHAPExplainer()
        logger.info("SHAP explainer loaded.")
    except Exception as e:
        logger.warning(f"Failed to load SHAP explainer: {e}")

    try:
        if os.path.exists(CACHE_DIR):
            _narratives_count = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    except Exception as e:
        logger.warning(f"Failed to count cached narratives: {e}")

def get_narratives_count() -> int:
    return _narratives_count

def get_shap_attribution(candidate_features: dict) -> dict:
    if not _explainer:
        return {"top_features": [], "baseline_score": 0.0}
    
    try:
        res = _explainer.explain_candidate(candidate_features)
        contributions = res["contributions"]
        
        sorted_feats = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]["shap_value"]), 
            reverse=True
        )
        
        top_features = []
        for feat, data in sorted_feats[:5]:
            val = float(data["raw_value"])
            shap_val = float(data["shap_value"])
            direction = "positive" if shap_val >= 0 else "negative"
            
            top_features.append(FeatureContribution(
                feature_name=feat,
                value=val,
                shap_contribution=shap_val,
                direction=direction
            ))
            
        return {
            "top_features": top_features,
            "baseline_score": res["expected_value"]
        }
    except Exception as e:
        logger.warning(f"SHAP attribution failed: {e}")
        return {"top_features": [], "baseline_score": 0.0}

def get_narrative(candidate_id: str, context: dict) -> tuple[str, bool, bool]:
    cache_path = os.path.join(CACHE_DIR, f"{candidate_id}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("narrative", ""), True, False
        except Exception as e:
            logger.warning(f"Failed to read narrative cache for {candidate_id}: {e}")
            
    fallback_text = generate_fallback_narrative(context)
    return fallback_text, False, True
