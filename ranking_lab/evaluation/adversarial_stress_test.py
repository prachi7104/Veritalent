"""
ranking_lab/evaluation/adversarial_stress_test.py

Injects synthetic adversarial profiles into the ranked list and verifies
the model correctly down-ranks them:

1. Keyword-stuffers: high skill_breadth and skill_mastery_triangulation,
   but zero tenure_stability and low product_vs_services.
2. Consistent-fraud (honeypots): high trust_score (risk=1.0), near-perfect
   on other features. The monotonic -1 constraint should suppress these.
3. Activity-fakers: high activity_quality_composite, zero skill_depth.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

# Helper to build a feature vector from a partial dict
def _make_vector(overrides: dict) -> np.ndarray:
    base = {f: 0.5 for f in TRAINING_FEATURES}
    base.update(overrides)
    return np.array([base[f] for f in TRAINING_FEATURES])


ADVERSARIAL_PROFILES = {
    "keyword_stuffer": _make_vector({
        "skill_breadth": 1.0,
        "skill_mastery_triangulation": 48.0,  # Max possible for 2 YOE expert (24 * 2 = 48)
        "skill_depth": 5.0,       # claims breadth but lacks depth
        "tenure_stability": 0.0,
        "product_vs_services": 0.1,
        "trust_score": 1.0,       # maximum risk (caught by new density check)
        "implied_skill_score": 0.0
    }),
    "consistent_fraud_honeypot": _make_vector({
        "trust_score": 1.0,        # maximum risk — monotonic -1 must suppress this
        "skill_depth": 100.0,
        "skill_breadth": 1.0,
        "tenure_stability": 50.0,
        "activity_quality_composite": 1.0,
        "product_vs_services": 1.0,
        "implied_skill_score": 0.0
    }),
    "activity_faker": _make_vector({
        "activity_quality_composite": 1.0,
        "skill_depth": 0.0,
        "skill_breadth": 0.0,
        "tenure_stability": 0.0,
        "trust_score": 0.3,
        "implied_skill_score": 0.0
    }),
}

# A reference "strong legitimate" candidate
STRONG_LEGITIMATE = _make_vector({
    "skill_depth": 80.0,
    "skill_breadth": 0.8,
    "skill_recency": 0.7,
    "skill_mastery_triangulation": 100.0,
    "tenure_stability": 40.0,
    "activity_quality_composite": 0.7,
    "trust_score": 0.1,          # low risk
    "logistics_fit_score": 0.8,
    "product_vs_services": 0.9,
    "implied_skill_score": 1.0
})


def run_adversarial_stress_test(model) -> dict:
    """
    Scores each adversarial profile vs. a strong legitimate candidate.
    Asserts that every adversarial profile is scored LOWER than the legitimate one.

    Returns:
      dict with per-profile scores, legitimate score, and per-profile PASS/FAIL verdict.
    """
    legit_score = float(model.predict(STRONG_LEGITIMATE.reshape(1, -1))[0])
    print(f"  [Legitimate candidate score]: {legit_score:.4f}")

    results = {"legitimate_score": legit_score, "profiles": {}}
    all_passed = True

    for name, vector in ADVERSARIAL_PROFILES.items():
        adv_score = float(model.predict(vector.reshape(1, -1))[0])
        passed = adv_score < legit_score
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}: score={adv_score:.4f} vs legit={legit_score:.4f}")
        results["profiles"][name] = {
            "score": adv_score,
            "passed": passed,
            "delta_vs_legit": adv_score - legit_score,
        }

    results["all_passed"] = all_passed
    results["verdict"] = "PASS" if all_passed else "FAIL — adversarial profiles not suppressed"
    return results

if __name__ == "__main__":
    from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
    
    # Update keyword_stuffer to have implied_skill_score=0.0
    # Because a real keyword stuffer wouldn't have narrative skills
    ADVERSARIAL_PROFILES["keyword_stuffer"][-1] = 0.0
    
    # Update legitimate candidate to have implied_skill_score=1.0
    STRONG_LEGITIMATE[-1] = 1.0
    
    model = GBMLambdaRankModel()
    model.load('ranking_lab/models/gbm_lambdarank.txt')
    run_adversarial_stress_test(model)

