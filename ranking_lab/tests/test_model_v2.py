import pytest
from pathlib import Path


def test_feature_config_v2_lengths_match():
    from ranking_lab.models.feature_config_v2 import FEATURE_NAMES_V2, MONOTONIC_CONSTRAINTS_V2
    assert len(FEATURE_NAMES_V2) == len(MONOTONIC_CONSTRAINTS_V2)


def test_jd_skill_score_has_positive_constraint():
    from ranking_lab.models.feature_config_v2 import FEATURE_NAMES_V2, MONOTONIC_CONSTRAINTS_V2
    m = dict(zip(FEATURE_NAMES_V2, MONOTONIC_CONSTRAINTS_V2))
    assert m["jd_skill_score"] == 1
    assert m["yoe_band_fit"] == 1
    assert m["trust_score"] == -1


def test_v2_model_file_exists_after_gate_pass():
    p = Path("ranking_lab/models/gbm_lambdarank_v2.txt")
    if not p.exists():
        pytest.skip("v2 model not trained yet")
    assert p.stat().st_size > 1000


def test_netflix_outranks_aganitha_on_jd_skill_score():
    """
    After Phase 1B, Netflix candidate (LtR+BM25+Weaviate = deep-IR)
    must outscore Aganitha candidate (QLoRA+PyTorch = buzzword).
    Validates the core claim of jd_skill_score.
    """
    from feature_lab.features.jd_skill_features import JDWeightedSkillScoreFeature
    feat = JDWeightedSkillScoreFeature()
    aganitha = {
        "profile": {"years_of_experience": 7.0}, "career_history": [],
        "skills": [
            {"name": "QLoRA", "proficiency": "advanced", "duration_months": 18, "endorsements": 0},
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 36, "endorsements": 2},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    netflix = {
        "profile": {"years_of_experience": 8.0}, "career_history": [],
        "skills": [
            {"name": "Learning to Rank", "proficiency": "expert", "duration_months": 36, "endorsements": 3},
            {"name": "BM25", "proficiency": "advanced", "duration_months": 24, "endorsements": 0},
            {"name": "Weaviate", "proficiency": "intermediate", "duration_months": 18, "endorsements": 0},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    s_aganitha, _ = feat.compute(aganitha)
    s_netflix, _  = feat.compute(netflix)
    assert s_netflix > s_aganitha, (
        f"Netflix (LtR+BM25+Weaviate) should beat Aganitha (QLoRA+PyTorch). "
        f"Got netflix={s_netflix:.2f}, aganitha={s_aganitha:.2f}"
    )
