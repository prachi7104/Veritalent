import pytest
from unittest.mock import patch

from backend.services.retrieval_service import _apply_hard_exclusions

@pytest.fixture
def mock_candidate_repository():
    dataset = {
        "CAND_1": {
            "profile": {"summary": "A seasoned Java developer.", "headline": "Java Backend Engineer"},
            "skills": [{"name": "Java"}, {"name": "Spring Boot"}]
        },
        "CAND_2": {
            "profile": {"summary": "I am a JavaScript developer.", "headline": "Frontend Dev"},
            "skills": [{"name": "JavaScript"}, {"name": "React"}]
        },
        "CAND_3": {
            "profile": {"summary": "Python and C++ expert", "headline": "Software Engineer"},
            "skills": [{"name": "Python"}, {"name": "C++"}, {"name": "Algorithms"}]
        }
    }
    
    with patch("backend.services.retrieval_service.get_candidate", side_effect=lambda cid: dataset.get(cid)):
        yield dataset

def test_empty_hard_exclusions(mock_candidate_repository):
    candidate_ids = ["CAND_1", "CAND_2", "CAND_3"]
    hard_exclusions = []
    
    result = _apply_hard_exclusions(candidate_ids, hard_exclusions, top_k=2)
    
    assert result == ["CAND_1", "CAND_2"]

def test_excluded_term_removes_candidate(mock_candidate_repository):
    candidate_ids = ["CAND_1", "CAND_2", "CAND_3"]
    hard_exclusions = ["Python"]
    
    result = _apply_hard_exclusions(candidate_ids, hard_exclusions, top_k=2)
    
    assert result == ["CAND_1", "CAND_2"]

def test_word_boundary_matching(mock_candidate_repository):
    candidate_ids = ["CAND_1", "CAND_2", "CAND_3"]
    hard_exclusions = ["Java"]
    
    result = _apply_hard_exclusions(candidate_ids, hard_exclusions, top_k=10)
    
    assert "CAND_1" not in result
    assert "CAND_2" in result
    assert "CAND_3" in result
    assert result == ["CAND_2", "CAND_3"]

def test_case_insensitive_matching(mock_candidate_repository):
    candidate_ids = ["CAND_1", "CAND_2", "CAND_3"]
    hard_exclusions = ["java"]
    
    result = _apply_hard_exclusions(candidate_ids, hard_exclusions, top_k=10)
    
    assert "CAND_1" not in result
    assert "CAND_2" in result
