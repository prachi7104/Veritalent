import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.scoring.ensemble_trust_score import compute_trust_score

def generate_adversarial_profile() -> dict:
    """
    Creates a perfectly consistent but fabricated profile.
    - YOE matches exactly the sum of career_history.
    - Expert skills have non-zero duration_months to bypass the check.
    - No assessment scores (so zero penalty from mismatch).
    - Email/Phone verified (bypasses identity checks).
    """
    profile = {
        "candidate_id": "CAND_ADVERSARIAL_001",
        "profile": {
            "years_of_experience": 10.0,
            "summary": "I am a Senior AI Engineer with extensive experience."
        },
        "career_history": [
            {
                "company": "FakeAI Corp",
                "title": "Senior AI Engineer",
                "duration_months": 120, # Exactly 10 years
                "description": "Led AI teams."
            }
        ],
        "skills": [
            {
                "name": "Machine Learning",
                "proficiency": "expert",
                "duration_months": 120, # Matches career history
                "endorsements": 50
            },
            {
                "name": "Deep Learning",
                "proficiency": "expert",
                "duration_months": 60,
                "endorsements": 20
            }
        ],
        "redrob_signals": {
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    return profile

def run_stress_test():
    adversarial_profile = generate_adversarial_profile()
    score_res = compute_trust_score(adversarial_profile)
    
    print("--- Adversarial Profile Stress Test ---")
    print(f"Trust Score: {score_res['trust_score']:.3f} (Lower = more trusted)")
    print("Breakdown:")
    print(json.dumps(score_res['breakdown'], indent=2))
    
    threshold = 0.4 # Typical review threshold
    if score_res['trust_score'] < threshold:
        print("\nRESULT: The adversarial profile PASSED undetected.")
        print("CONCLUSION: The current trust score catches sloppy/inconsistent fraud; it does NOT reliably catch sophisticated, internally-consistent fraud.")
    else:
        print("\nRESULT: The adversarial profile was CAUGHT.")

if __name__ == "__main__":
    run_stress_test()
