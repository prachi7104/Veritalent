import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trust_lab.evaluation.bias_analysis import compute_career_gaps, compute_industry_switches, is_technical_study

class TestBiasAnalysis(unittest.TestCase):
    def test_compute_career_gaps(self):
        history = [
            {"start_date": "2015-01-01", "end_date": "2016-01-01"},
            {"start_date": "2016-02-01", "end_date": "2017-01-01"}, # Gap < 90 days
            {"start_date": "2018-01-01", "end_date": "2019-01-01"}  # Gap ~1 year
        ]
        self.assertEqual(compute_career_gaps(history), 1)
        
    def test_compute_industry_switches(self):
        history = [
            {"start_date": "2015-01-01", "industry": "Tech"},
            {"start_date": "2016-02-01", "industry": "Tech"},
            {"start_date": "2018-01-01", "industry": "Finance"} 
        ]
        self.assertEqual(compute_industry_switches(history), 1)

    def test_is_technical_study(self):
        self.assertTrue(is_technical_study([{"field_of_study": "Computer Science"}]))
        self.assertFalse(is_technical_study([{"field_of_study": "Business Administration"}]))

if __name__ == '__main__':
    unittest.main()
