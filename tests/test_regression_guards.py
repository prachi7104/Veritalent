# tests/test_regression_guards.py
"""
Regression guards for Veritalent v2 pipeline.

These tests encode every number that was proven by experiment.
If any fails after a code change, the change broke a validated invariant.
All assertions reference the experiment that produced the number.
"""
import pytest
import json
import numpy as np
from pathlib import Path


# ── Feature design guards ─────────────────────────────────────────────────────

def test_jd_skill_score_deep_ir_3x_buzzword():
    """
    deep-IR skills must score 3× more than buzzword skills at same duration/proficiency.
    Evidence: JD_BAND_WEIGHTS = {deep-ir: 3.0, buzzword: 1.0} in jd_skill_features.py
    Regression: any change to JD_BAND_WEIGHTS.
    """
    from feature_lab.features.jd_skill_features import JDWeightedSkillScoreFeature
    feat = JDWeightedSkillScoreFeature()
    deep = {
        "profile": {}, "career_history": [],
        "skills": [{"name": "Elasticsearch", "proficiency": "expert",
                    "duration_months": 12, "endorsements": 0}],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    buzz = {
        "profile": {}, "career_history": [],
        "skills": [{"name": "LangChain", "proficiency": "expert",
                    "duration_months": 12, "endorsements": 0}],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    s_deep, _ = feat.compute(deep)
    s_buzz, _ = feat.compute(buzz)
    assert abs(s_deep / s_buzz - 3.0) < 0.1, (
        f"deep-IR:buzzword ratio = {s_deep/s_buzz:.2f}, expected ~3.0. "
        "Check JD_BAND_WEIGHTS."
    )


def test_netflix_outranks_aganitha_on_jd_skill_score():
    """
    Netflix candidate (LtR+BM25+Weaviate = deep-IR) must score higher than
    Aganitha (QLoRA+PyTorch = buzzword) on jd_skill_score.
    Evidence: Fix-3 spot-check confirmed Netflix rank 46, Aganitha rank 108.
    This is the core correctness proof for JD-weighted ranking.
    Regression: any change to skill band assignments.
    """
    from feature_lab.features.jd_skill_features import JDWeightedSkillScoreFeature
    feat = JDWeightedSkillScoreFeature()
    netflix = {
        "profile": {"years_of_experience": 8.0}, "career_history": [],
        "skills": [
            {"name": "Learning to Rank", "proficiency": "expert",
             "duration_months": 36, "endorsements": 3},
            {"name": "BM25", "proficiency": "advanced",
             "duration_months": 24, "endorsements": 0},
            {"name": "Weaviate", "proficiency": "intermediate",
             "duration_months": 18, "endorsements": 0},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    aganitha = {
        "profile": {"years_of_experience": 7.0}, "career_history": [],
        "skills": [
            {"name": "QLoRA", "proficiency": "advanced",
             "duration_months": 18, "endorsements": 0},
            {"name": "PyTorch", "proficiency": "expert",
             "duration_months": 36, "endorsements": 2},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    s_n, _ = feat.compute(netflix)
    s_a, _ = feat.compute(aganitha)
    assert s_n > s_a, (
        f"Netflix (LtR+BM25+Weaviate={s_n:.1f}) must beat "
        f"Aganitha (QLoRA+PyTorch={s_a:.1f}). "
        "Skill band assignments may have regressed."
    )


def test_product_vs_services_catches_all_tier1_firms():
    """
    All original 9 Tier-1 firms plus key Tier-2 additions must be classified as services.
    Evidence: PROMPT_01D expanded list from 9→50+ firms, mean 0.838→0.689.
    Regression: removing firms from SERVICES_FIRMS.
    """
    from feature_lab.features.company_features import _is_services_firm
    must_catch = [
        # Original 9
        "TCS", "Infosys", "Wipro", "Cognizant", "Accenture",
        "Capgemini", "HCL", "Tech Mahindra", "Mindtree",
        # Key Tier-2 additions from PROMPT_01D
        "Mphasis", "Hexaware Technologies", "Persistent Systems",
        "L&T Infotech", "Zensar Technologies", "LTIMindtree",
    ]
    failures = [f for f in must_catch if not _is_services_firm(f)]
    assert not failures, (
        f"These firms should be classified as services but aren't: {failures}. "
        "Check SERVICES_FIRMS list in company_features.py."
    )


def test_product_vs_services_pool_mean_below_075():
    """
    After firm list expansion, pool mean must stay below 0.75.
    Baseline was 0.838 (too high). PROMPT_01D brought it to 0.689.
    Regression: removing firms raises mean back toward 0.838.
    """
    store_path = Path("feature_lab/store/feature_store.jsonl")
    if not store_path.exists():
        pytest.skip("feature_store.jsonl not built")
    vals = []
    with open(store_path, encoding="utf-8") as f:
        # Check first 500 rows for subset speed
        for i, line in enumerate(f):
            if i >= 500: break
            row = json.loads(line)
            v = row.get("product_vs_services")
            if v is not None:
                vals.append(float(v))
    mean = np.mean(vals)
    assert mean < 0.75, (
        f"product_vs_services pool mean = {mean:.4f} >= 0.75. "
        "Firm list may have regressed. PROMPT_01D target was 0.689."
    )


# ── Blend scoring guards ──────────────────────────────────────────────────────

def test_blend_alpha_is_0_90():
    """
    alpha must be 0.90 — proven optimal in Fix-3 alpha sweep.
    alpha=0.85 gave NDCG=-0.0036 regression. alpha=0.90 gave +0.0009.
    Regression: changing alpha without re-running the sweep.
    """
    p = Path("ranking_lab/models/blend_config.json")
    if not p.exists():
        pytest.skip("blend_config.json not found")
    with open(p) as f:
        cfg = json.load(f)
    assert abs(cfg["alpha"] - 0.90) < 0.001, (
        f"alpha={cfg['alpha']} — expected 0.90. "
        "Re-run exp_f_fix3_blend.py alpha sweep before changing."
    )


def test_blend_yoe_weight_is_zero():
    """
    yoe_band_fit weight must be 0.0 — ablation G3 vs G4 = 0.0000 marginal.
    Evidence: PROMPT_03A experiment G3 proved zero contribution.
    Regression: restoring yoe_band_fit to blend without re-running ablation.
    """
    p = Path("ranking_lab/models/blend_config.json")
    if not p.exists():
        pytest.skip("blend_config.json not found")
    with open(p) as f:
        cfg = json.load(f)
    assert cfg["new_features"]["yoe_band_fit"] == 0.0, (
        "yoe_band_fit must be 0.0. Ablation G3 proved zero marginal contribution. "
        "Re-run PROMPT_03A G3 before restoring."
    )


def test_blend_jd_weight_is_10_percent():
    """
    jd_skill_score takes the full 10% non-GBM slice.
    Evidence: ablation G2 vs G4 = +0.0072 marginal contribution.
    """
    p = Path("ranking_lab/models/blend_config.json")
    if not p.exists():
        pytest.skip("blend_config.json not found")
    with open(p) as f:
        cfg = json.load(f)
    jd_w    = cfg["new_features"]["jd_skill_score"]
    alpha   = cfg["alpha"]
    # Effective weight = (1-alpha) × jd_w = 0.10 × 1.0 = 0.10
    effective = (1 - alpha) * jd_w
    assert abs(effective - 0.10) < 0.005, (
        f"Effective jd_skill weight = {effective:.3f}, expected 0.10. "
        f"alpha={alpha}, jd_w={jd_w}."
    )


def test_no_gbm_feature_dominates_over_50pct():
    """
    No single feature in the base GBM should have > 50% of total importance.
    Evidence: PROMPT_03A audit showed skill_mastery at 28.4% (healthy).
    Regression: model replacement with a less-balanced model.
    """
    from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES
    model = GBMLambdaRankModel()
    model.load("ranking_lab/models/gbm_lambdarank.txt")
    # model.model is a raw lightgbm.Booster after load
    imps = model.model.feature_importance()
    total = sum(imps)
    top = max(imps)
    assert top / total < 0.50, (
        f"Top GBM feature has {top/total:.1%} importance — should be < 50%. "
        "Was 28.4% (skill_mastery_triangulation) in PROMPT_03A audit."
    )


# ── Narrative quality guards ──────────────────────────────────────────────────

def test_shap_formatter_covers_all_training_features():
    """
    SHAP formatter must have a human-readable label for every TRAINING_FEATURE.
    Regression: adding a new feature without adding its translation.
    """
    from explainability_lab.narrative.shap_formatter import FEATURE_LABELS
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES
    missing = [f for f in TRAINING_FEATURES if f not in FEATURE_LABELS]
    assert not missing, (
        f"These training features have no SHAP label translation: {missing}. "
        "Add them to FEATURE_LABELS in shap_formatter.py."
    )


def test_submission_narratives_no_raw_feature_names():
    """
    submission_v2.csv reasoning must contain no raw feature names.
    Evidence: PROMPT_01E + PROMPT_03_FIX_C both enforced this.
    Checks top-20 rows only (sufficient signal).
    """
    import pandas as pd
    p = Path("submission/submission_v2.csv")
    if not p.exists():
        pytest.skip("submission_v2.csv not built")
    df = pd.read_csv(p)
    raw = [
        "skill_mastery_triangulation", "skill_depth", "skill_breadth",
        "activity_quality_composite", "tenure_stability",
        "logistics_fit_score", "product_vs_services",
        "implied_skill_score", "jd_skill_score", "yoe_band_fit",
    ]
    for _, row in df.head(20).iterrows():
        found = [f for f in raw if f in str(row["reasoning"])]
        assert not found, (
            f"Rank {row['rank']} reasoning contains raw feature names: {found}"
        )


# ── Submission format guards ──────────────────────────────────────────────────

def test_submission_cand_0037980_is_rank_1():
    """
    CAND_0037980 must be rank 1 in final submission.
    Evidence: was mastery_rank=89, jd_skill_rank=1 (PROMPT_01B).
    Now confirmed rank 1 in submission_v2.csv (PROMPT_04 output).
    This is the single strongest proof that jd_skill_score corrected the ranking.
    Regression: any change that drops this candidate from rank 1.
    """
    import pandas as pd
    p = Path("submission/submission_v2.csv")
    if not p.exists():
        pytest.skip("submission_v2.csv not built")
    df = pd.read_csv(p)
    rank1_cid = df[df["rank"] == 1]["candidate_id"].iloc[0]
    assert rank1_cid == "CAND_0037980", (
        f"Expected CAND_0037980 at rank 1, got {rank1_cid}. "
        "This candidate jumped from mastery_rank=89 to jd_skill_rank=1. "
        "If rank 1 changed, the blend scoring may have regressed."
    )


def test_submission_score_range_is_normalized():
    """
    Blend scores must be in (0, 1] range — a consequence of normalization.
    Evidence: PROMPT_04 output showed range 0.8213-0.9978.
    Regression: scoring pipeline bypassing the normalization step.
    """
    import pandas as pd
    p = Path("submission/submission_v2.csv")
    if not p.exists():
        pytest.skip("submission_v2.csv not built")
    df = pd.read_csv(p)
    assert df["score"].min() > 0.0, "Scores should be positive after normalization"
    assert df["score"].max() <= 1.0, "Scores should be <= 1.0 after normalization"
    assert df["score"].max() > 0.95, (
        "Top score should be > 0.95. If lower, normalization may be broken."
    )
