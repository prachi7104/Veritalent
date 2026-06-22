from typing import Dict, Type, List, Any, Tuple
from .base import BaseFeature

class FeatureRegistry:
    """
    Central registry for all feature extractors.
    """
    def __init__(self):
        self._features: Dict[str, BaseFeature] = {}

    def register(self, feature: BaseFeature):
        """Register an instantiated feature."""
        if feature.name in self._features:
            raise ValueError(f"Feature with name '{feature.name}' already registered.")
        self._features[feature.name] = feature

    def get_feature(self, name: str) -> BaseFeature:
        """Retrieve a registered feature by name."""
        return self._features[name]

    def get_all_features(self) -> List[BaseFeature]:
        """Retrieve all registered features."""
        return list(self._features.values())

    def compute_all(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all registered features for a candidate.
        Returns a dict of feature outputs and reliability tags.
        """
        results = {}
        for feat in self._features.values():
            val, tag = feat.compute(candidate)
            
            # Flatten dict outputs (e.g. from trust_score stub)
            if isinstance(val, dict):
                for k, v in val.items():
                    results[k] = v
            else:
                results[feat.name] = val
                
            results[f"{feat.name}_reliability"] = tag
        return results

# Global registry instance
registry = FeatureRegistry()
