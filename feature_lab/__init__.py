"""
Feature Engineering Lab — __init__.py
Imports all feature modules to ensure their registry.register() calls execute,
populating the global registry singleton before the store reads from it.
"""

# Explicit ordering: base first, registry second, then all feature modules
from .features import base
from .features import feature_registry

# Each of these imports triggers module-level registry.register() calls
from .features import skill_features
from .features import career_features
from .features import trust_features
from .features import activity_features
from .features import industry_features
from .features import logistics_features
from .features import company_features
from .features import implied_skill_features
