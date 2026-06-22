from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseFeature(ABC):
    """
    Base class for all candidate features.
    
    Attributes:
        name: The unique string identifier for the feature in the store.
        version: Integer version of the feature logic.
        default_reliability_tag: One of 'clean', 'sparse', or 'experimental'.
    """
    def __init__(self, name: str, version: int, default_reliability_tag: str):
        if default_reliability_tag not in ["clean", "sparse", "experimental"]:
            raise ValueError(f"Invalid reliability_tag: {default_reliability_tag}")
        self.name = name
        self.version = version
        self.default_reliability_tag = default_reliability_tag

    @abstractmethod
    def compute(self, candidate: Dict[str, Any]) -> Tuple[Any, str]:
        """
        Compute the feature value given the raw candidate dictionary.
        Must return a tuple of (computed_value, reliability_tag).
        If the feature cannot be computed (e.g., missing data), return (default_value, tag).
        """
        pass
