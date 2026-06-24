import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from feature_lab.features.implied_skill_features import ImpliedSkillFeatures

def test_single_match():
    feat = ImpliedSkillFeatures()
    cand = {
        "profile": {
            "summary": "built production ranking pipeline for search results"
        }
    }
    out, tag = feat.compute(cand)
    assert tag == "clean"
    assert out["implied_skill_score"] == 1.0 / 5.0
    assert "ranking_systems" in out["implied_skill_categories"]

def test_no_match():
    feat = ImpliedSkillFeatures()
    cand = {
        "profile": {
            "summary": "I am a software engineer with experience in Python"
        }
    }
    out, tag = feat.compute(cand)
    assert out["implied_skill_score"] == 0.0
    assert out["implied_skill_categories"] == ""

def test_multiple_match_headline():
    feat = ImpliedSkillFeatures()
    cand = {
        "profile": {
            "headline": "Senior IR Engineer | Search Ranking | BM25 | Semantic Search"
        }
    }
    out, tag = feat.compute(cand)
    assert out["implied_skill_score"] >= 2.0 / 5.0
    assert "ranking_systems" in out["implied_skill_categories"]
    assert "vector_search" in out["implied_skill_categories"]
    assert "|" in out["implied_skill_categories"]

def test_case_insensitive():
    feat = ImpliedSkillFeatures()
    cand1 = {"profile": {"summary": "Ranking Pipeline"}}
    cand2 = {"profile": {"summary": "ranking pipeline"}}
    
    out1, _ = feat.compute(cand1)
    out2, _ = feat.compute(cand2)
    assert out1["implied_skill_score"] == out2["implied_skill_score"]
    assert out1["implied_skill_score"] > 0

def test_empty_narrative():
    feat = ImpliedSkillFeatures()
    out1, _ = feat.compute({"profile": {"summary": None}})
    out2, _ = feat.compute({"profile": {"summary": ""}})
    out3, _ = feat.compute({})
    
    assert out1["implied_skill_score"] == 0.0
    assert out2["implied_skill_score"] == 0.0
    assert out3["implied_skill_score"] == 0.0
