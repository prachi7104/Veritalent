import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.scoring.ensemble_trust_score import compute_trust_score

class TestEnsembleTrustScore(unittest.TestCase):
    def test_zero_assessment_coverage(self):
        cand = {
            "profile": {"years_of_experience": 5.0},
            "career_history": [{"duration_months": 60}],
            "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 36, "endorsements": 5}],
            "redrob_signals": {
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            } # No skill_assessment_scores
        }
        res = compute_trust_score(cand)
        self.assertEqual(res["breakdown"]["assm_risk"], 0.0)
        # Verify it doesn't penalize
        self.assertLess(res["trust_score"], 0.2)
        
    def test_full_template_reliance_but_consistent(self):
        cand = {
            "profile": {
                "years_of_experience": 5.0,
                "summary": "This is a known template string."
            },
            "career_history": [
                {"duration_months": 60, "description": "This is a known template string."}
            ],
            "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 36, "endorsements": 5}],
            "redrob_signals": {
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        }
        known_templates = {"This is a known template string."}
        res = compute_trust_score(cand, known_templates=known_templates)
        # Template fraction is 1.0, but with low weight (0.2 out of ~3.7 total weight)
        # Total weight = 1(yoe) + 1(prof) + 0(assm) + 0.2(temp) + 0.5(ident) = 2.7
        # Total risk = 0 + 0 + 0 + (1.0 * 0.2) + 0 = 0.2
        # Score = 0.2 / 2.7 = 0.074
        self.assertLess(res["trust_score"], 0.2)
        self.assertEqual(res["breakdown"]["temp_risk"], 1.0)

    def test_auto_reject_absence(self):
        # We verify that no "auto_reject" flag exists in the returned dictionary.
        # This confirms that downstream components must implement any gating logic.
        cand = {}
        res = compute_trust_score(cand)
        self.assertNotIn("auto_reject", res)
        self.assertNotIn("rejected", res)

if __name__ == '__main__':
    unittest.main()
