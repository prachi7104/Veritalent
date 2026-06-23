import math

class SchemaViolationError(Exception):
    """Raised when an architectural constraint from the master context is violated."""
    pass

class GuardedEducationDict(dict):
    def __getitem__(self, key):
        if key in ("start_year", "end_year"):
            raise SchemaViolationError(
                f"ILLEGAL ACCESS: Attempted to access education.{key}. "
                "The master context explicitly forbids using education date fields for anomaly detection "
                "due to unreliability."
            )
        return super().__getitem__(key)
        
    def get(self, key, default=None):
        if key in ("start_year", "end_year"):
            raise SchemaViolationError(
                f"ILLEGAL ACCESS: Attempted to access education.{key} via get(). "
                "The master context explicitly forbids using education date fields."
            )
        return super().get(key, default)

def enforce_schema(candidate: dict) -> dict:
    """
    Enforces master context constraints on a raw candidate dictionary.
    Modifies the dictionary in place and returns it.
    """
    # 1. Enforce -1 sentinels for github_activity_score and offer_acceptance_rate
    signals = candidate.get("redrob_signals", {})
    
    if signals.get("github_activity_score") == -1:
        signals["github_activity_score"] = math.nan
        
    if signals.get("offer_acceptance_rate") == -1:
        signals["offer_acceptance_rate"] = math.nan
        
    # 2. Enforce education dates constraint
    if "education" in candidate:
        guarded_education = []
        for edu in candidate["education"]:
            guarded_edu = GuardedEducationDict(edu)
            guarded_education.append(guarded_edu)
        candidate["education"] = guarded_education

    return candidate
