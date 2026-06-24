import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trust_lab.checks.skill_density_check import check_skill_density
import trust_lab.checks.skill_density_check as mod

def test_high_density_max_suspicion():
    # 15 skills, 2 years => 15 / (2 * 3.0) = 2.5 => capped at 1.0
    cand = {
        "profile": {"years_of_experience": 2.0},
        "skills": [
            {"name": "pytorch"}, {"name": "tensorflow"}, {"name": "nlp"},
            {"name": "machine learning"}, {"name": "deep learning"}, {"name": "bm25"},
            {"name": "learning to rank"}, {"name": "qdrant"}, {"name": "weaviate"},
            {"name": "milvus"}, {"name": "scikit-learn"}, {"name": "elasticsearch"},
            {"name": "opensearch"}, {"name": "llamaindex"}, {"name": "haystack"}
        ]
    }
    res = check_skill_density(cand)
    assert res["keyword_stuffing_density"] == 1.0

def test_legit_senior_low_suspicion():
    # 8 skills, 7 years => 8 / (7 * 3.0) = 0.38
    cand = {
        "profile": {"years_of_experience": 7.0},
        "skills": [
            {"name": "pytorch"}, {"name": "tensorflow"}, {"name": "nlp"},
            {"name": "machine learning"}, {"name": "deep learning"}, {"name": "bm25"},
            {"name": "rag"}, {"name": "langchain"}
        ]
    }
    res = check_skill_density(cand)
    assert 0.38 < res["keyword_stuffing_density"] < 0.39

def test_zero_skills():
    cand = {
        "profile": {"years_of_experience": 5.0},
        "skills": []
    }
    res = check_skill_density(cand)
    assert res["keyword_stuffing_density"] == 0.0

def test_zero_experience():
    cand = {
        "profile": {"years_of_experience": 0.0},
        "skills": [{"name": "pytorch"}]
    }
    res = check_skill_density(cand)
    assert res["keyword_stuffing_density"] == 1.0

def test_configurable_threshold(monkeypatch):
    monkeypatch.setattr(mod, "SKILL_DENSITY_THRESHOLD", 1.0)
    # 8 skills, 7 years => 8 / (7 * 1.0) = 1.14 => capped at 1.0
    cand = {
        "profile": {"years_of_experience": 7.0},
        "skills": [
            {"name": "pytorch"}, {"name": "tensorflow"}, {"name": "nlp"},
            {"name": "machine learning"}, {"name": "deep learning"}, {"name": "bm25"},
            {"name": "rag"}, {"name": "langchain"}
        ]
    }
    res = check_skill_density(cand)
    assert res["keyword_stuffing_density"] == 1.0
