"""
test_frequency_audit.py — Tests for the independent frequency re-derivation.
"""
import json
import math
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fingerprint_lab.analysis.frequency_audit import (
    run_frequency_audit, is_senior_ai_ml_title, CLAIMED_FINGERPRINT_SKILLS
)


def _make_jsonl(candidates: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")


class TestSeniorAiMlTitleDetection:
    def test_senior_ml_engineer_is_detected(self):
        assert is_senior_ai_ml_title("Senior ML Engineer")

    def test_staff_machine_learning_is_detected(self):
        assert is_senior_ai_ml_title("Staff Machine Learning Engineer")

    def test_lead_ai_engineer_is_detected(self):
        assert is_senior_ai_ml_title("Lead AI Engineer")

    def test_junior_ml_not_senior(self):
        assert not is_senior_ai_ml_title("Junior ML Engineer")

    def test_senior_software_engineer_no_ai_ml(self):
        assert not is_senior_ai_ml_title("Senior Software Engineer")

    def test_data_scientist_with_no_seniority(self):
        assert not is_senior_ai_ml_title("Data Scientist")

    def test_senior_data_scientist_detected(self):
        assert is_senior_ai_ml_title("Senior Data Scientist")


class TestFrequencyAuditSmallSyntheticExample:
    """
    Verify the audit correctly identifies a known rare-skill pattern
    in a small synthetic dataset.
    """

    def test_rare_skill_identified(self, tmp_path):
        # Plant 3 occurrences of "Search Backend" (a fingerprint skill) in 20 candidates
        candidates = []
        for i in range(20):
            c = {
                "candidate_id": f"CAND_{i:07d}",
                "profile": {
                    "current_title": "Senior ML Engineer" if i < 3 else "Software Engineer",
                },
                "skills": [],
                "redrob_signals": {},
                "career_history": [],
                "education": [],
            }
            if i < 3:
                c["skills"].append({
                    "name": "Search Backend",
                    "proficiency": "expert",
                    "duration_months": 24,
                    "endorsements": 5,
                })
            else:
                c["skills"].append({
                    "name": "Python",
                    "proficiency": "intermediate",
                    "duration_months": 12,
                    "endorsements": 0,
                })
            candidates.append(c)

        jsonl_path = str(tmp_path / "test_candidates.jsonl")
        _make_jsonl(candidates, jsonl_path)

        results = run_frequency_audit(jsonl_path, None)

        assert results["total_candidates_scanned"] == 20
        assert "Search Backend" in results["fingerprint_skills_found"]
        # Only 3 of the 13 claimed FP skills appear -> 10 should be absent
        assert len(results["fingerprint_skills_absent"]) == 12
        assert results["total_fp_holders"] == 3
        assert results["seniority_alignment_rate"] == 1.0  # all 3 are Senior ML
        assert results["senior_ai_ml_count"] == 3

    def test_absent_skill_flagged_correctly(self, tmp_path):
        """If a fingerprint skill is completely absent, it must be in fingerprint_skills_absent."""
        candidates = [
            {
                "candidate_id": "CAND_0000001",
                "profile": {"current_title": "Software Engineer"},
                "skills": [{"name": "Python", "proficiency": "expert",
                             "duration_months": 12, "endorsements": 0}],
                "redrob_signals": {}, "career_history": [], "education": [],
            }
        ]
        jsonl_path = str(tmp_path / "test_absent.jsonl")
        _make_jsonl(candidates, jsonl_path)
        results = run_frequency_audit(jsonl_path, None)
        # All 13 fingerprint skills should be absent
        assert len(results["fingerprint_skills_absent"]) == 13
        assert results["total_fp_holders"] == 0

    def test_non_senior_holder_lowers_alignment(self, tmp_path):
        """If a non-senior candidate holds a fingerprint skill, alignment < 1.0."""
        candidates = [
            {
                "candidate_id": "CAND_0000001",
                "profile": {"current_title": "Senior ML Engineer"},
                "skills": [{"name": "Search Backend", "proficiency": "expert",
                             "duration_months": 24, "endorsements": 5}],
                "redrob_signals": {}, "career_history": [], "education": [],
            },
            {
                "candidate_id": "CAND_0000002",
                "profile": {"current_title": "Junior Software Engineer"},  # NOT senior AI/ML
                "skills": [{"name": "Search Backend", "proficiency": "intermediate",
                             "duration_months": 6, "endorsements": 0}],
                "redrob_signals": {}, "career_history": [], "education": [],
            },
        ]
        jsonl_path = str(tmp_path / "test_alignment.jsonl")
        _make_jsonl(candidates, jsonl_path)
        results = run_frequency_audit(jsonl_path, None)
        assert results["total_fp_holders"] == 2
        assert results["seniority_alignment_rate"] == 0.5  # 1 of 2 is senior AI/ML

    def test_output_json_written_correctly(self, tmp_path):
        candidates = [
            {
                "candidate_id": "CAND_0000001",
                "profile": {"current_title": "Lead AI Engineer"},
                "skills": [{"name": "Ranking Systems", "proficiency": "advanced",
                             "duration_months": 18, "endorsements": 3}],
                "redrob_signals": {}, "career_history": [], "education": [],
            }
        ]
        jsonl_path = str(tmp_path / "test_output.jsonl")
        out_path   = str(tmp_path / "audit_out.json")
        _make_jsonl(candidates, jsonl_path)
        run_frequency_audit(jsonl_path, out_path)
        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            data = json.load(f)
        assert data["total_fp_holders"] == 1
        assert "Ranking Systems" in data["fingerprint_skills_found"]


class TestMarginalValueNdcg:
    """Verify NDCG computation against a hand-checked toy case."""

    def test_perfect_ranking(self):
        from fingerprint_lab.analysis.marginal_value_test import compute_ndcg_at_k
        # 3 relevant candidates ranked 1, 2, 3 => NDCG = 1.0
        ranked = ["A", "B", "C", "D", "E"]
        judgments = {"A": 3, "B": 2, "C": 1, "D": 0, "E": 0}
        ndcg = compute_ndcg_at_k(ranked, judgments, k=5)
        assert abs(ndcg - 1.0) < 1e-6

    def test_reversed_ranking_is_lower(self):
        from fingerprint_lab.analysis.marginal_value_test import compute_ndcg_at_k
        ranked_good = ["A", "B", "C"]
        ranked_bad  = ["C", "B", "A"]
        judgments = {"A": 3, "B": 1, "C": 0}
        ndcg_good = compute_ndcg_at_k(ranked_good, judgments, k=3)
        ndcg_bad  = compute_ndcg_at_k(ranked_bad, judgments, k=3)
        assert ndcg_good > ndcg_bad

    def test_no_relevant_docs_is_zero(self):
        from fingerprint_lab.analysis.marginal_value_test import compute_ndcg_at_k
        ranked = ["X", "Y", "Z"]
        judgments = {"X": 0, "Y": 0, "Z": 0}
        ndcg = compute_ndcg_at_k(ranked, judgments, k=3)
        assert ndcg == 0.0

    def test_fp_bonus_moves_holders_up(self):
        """If a candidate has a fingerprint bonus, they should rank higher."""
        from fingerprint_lab.analysis.marginal_value_test import compute_ndcg_at_k
        # Simulate: FP holder is at rank 60 without bonus, moves to rank 5 with bonus
        # NDCG@50 should improve
        ranked_without = [f"X{i}" for i in range(100)]
        fp_id = "FP_HOLDER"
        ranked_without.insert(60, fp_id)  # rank 61 without bonus

        ranked_with = [fp_id] + [f"X{i}" for i in range(100)]  # rank 1 with bonus

        judgments = {fp_id: 3}  # only the fp holder is relevant

        ndcg_without = compute_ndcg_at_k(ranked_without, judgments, k=50)
        ndcg_with    = compute_ndcg_at_k(ranked_with, judgments, k=50)
        assert ndcg_with > ndcg_without
