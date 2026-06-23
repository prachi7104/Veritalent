import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.checks.proficiency_plausibility import check_proficiency_plausibility

class TestProficiencyPlausibility(unittest.TestCase):
    def test_plausible(self):
        cand = {
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 60, "endorsements": 10}
            ]
        }
        res = check_proficiency_plausibility(cand)
        self.assertEqual(res["implausible_count"], 0)
        self.assertEqual(res["severity_sum"], 0.0)
        self.assertFalse(res["flagged_honeypot_pattern"])

    def test_implausible_expert(self):
        cand = {
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
            ]
        }
        res = check_proficiency_plausibility(cand)
        self.assertEqual(res["implausible_count"], 1)
        self.assertEqual(res["severity_sum"], 1.0)
        self.assertFalse(res["flagged_honeypot_pattern"]) # Needs 3 for honeypot pattern

    def test_honeypot_pattern(self):
        cand = {
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
                {"name": "Java", "proficiency": "expert", "duration_months": 0, "endorsements": 5}, # Still 0 months, triggers honeypot flag but not the severity rule which also checks endorsements=0
                {"name": "C++", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
            ]
        }
        res = check_proficiency_plausibility(cand)
        # 2 have both 0 duration and 0 endorsements
        self.assertEqual(res["implausible_count"], 2)
        # All 3 have expert and 0 duration, so honeypot pattern is True
        self.assertTrue(res["flagged_honeypot_pattern"])

if __name__ == '__main__':
    unittest.main()
