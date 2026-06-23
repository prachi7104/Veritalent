import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.checks.yoe_tenure_consistency import check_yoe_consistency

class TestYoeConsistency(unittest.TestCase):
    def test_consistent(self):
        cand = {
            "profile": {"years_of_experience": 5.0},
            "career_history": [
                {"duration_months": 60}
            ]
        }
        res = check_yoe_consistency(cand)
        self.assertFalse(res["flagged"])
        self.assertEqual(res["deviation_years"], 0.0)
        
    def test_inconsistent(self):
        cand = {
            "profile": {"years_of_experience": 10.0},
            "career_history": [
                {"duration_months": 60} # 5 years
            ]
        }
        res = check_yoe_consistency(cand)
        self.assertTrue(res["flagged"])
        self.assertEqual(res["deviation_years"], 5.0)

    def test_multi_year_gap_legitimate(self):
        # A candidate with a multi-year gap might still be consistent if YOE accounts for actual work.
        cand = {
            "profile": {"years_of_experience": 5.0}, # Claims 5 years of experience
            "career_history": [
                {"duration_months": 30},
                # e.g., 2 year gap here
                {"duration_months": 30}
            ]
        }
        res = check_yoe_consistency(cand)
        self.assertFalse(res["flagged"])
        self.assertEqual(res["deviation_years"], 0.0)

if __name__ == '__main__':
    unittest.main()
