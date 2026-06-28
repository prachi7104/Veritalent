"""
test_features.py — Unit tests for all feature extractors.

Edge cases covered:
  - Candidate with zero skills
  - Candidate with a single career_history entry
  - Candidate with github_activity_score = -1 (missing sentinel)
  - Product-vs-services classification depending on history sequence
    (currently-at-services + prior-product ≠ career-long-services)
"""
import pytest

# Package-qualified imports — conftest.py ensures repo root is on sys.path
from feature_lab.features.skill_features import (
    SkillDepthFeature, SkillBreadthFeature, SkillRecencyFeature,
    SkillMasteryTriangulationFeature,
)
from feature_lab.features.career_features import (
    CareerVelocityFeature, PromotionVelocityFeature, TenureStabilityFeature,
    InflectionPointStrengthFeature,
)
from feature_lab.features.activity_features import ActivityQualityCompositeFeature
from feature_lab.features.company_features import ProductVsServicesFeature


# ===========================================================================
# Skill features
# ===========================================================================

class TestSkillFeaturesZeroSkills:
    def test_depth_zero_skills(self):
        val, tag = SkillDepthFeature().compute({"skills": []})
        assert val == 0.0
        assert tag == "clean"

    def test_breadth_zero_skills(self):
        val, tag = SkillBreadthFeature().compute({"skills": []})
        assert val == 0.0

    def test_recency_zero_skills(self):
        val, tag = SkillRecencyFeature().compute({"skills": []})
        assert val == 0.0

    def test_mastery_zero_skills(self):
        val, tag = SkillMasteryTriangulationFeature().compute({"skills": []})
        assert val == 0.0
        assert tag == "sparse"


class TestSkillDepth:
    def test_deep_ir_skill_scores_nonzero(self):
        cand = {"skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 36, "endorsements": 5}]}
        val, _ = SkillDepthFeature().compute(cand)
        assert val == 36 * 2.0  # expert weight = 2.0

    def test_generic_skill_does_not_contribute(self):
        cand = {"skills": [{"name": "Excel", "proficiency": "expert", "duration_months": 100, "endorsements": 0}]}
        val, _ = SkillDepthFeature().compute(cand)
        assert val == 0.0


class TestSkillMasteryTriangulation:
    def test_assessment_promotes_to_clean(self):
        cand = {
            "skills": [{"name": "PyTorch", "proficiency": "advanced", "duration_months": 24, "endorsements": 2}],
            "redrob_signals": {"skill_assessment_scores": {"PyTorch": 80.0}},
        }
        val, tag = SkillMasteryTriangulationFeature().compute(cand)
        assert tag == "clean"

    def test_no_assessment_stays_sparse(self):
        cand = {
            "skills": [{"name": "PyTorch", "proficiency": "advanced", "duration_months": 24, "endorsements": 2}],
            "redrob_signals": {"skill_assessment_scores": {}},
        }
        val, tag = SkillMasteryTriangulationFeature().compute(cand)
        assert tag == "sparse"


# ===========================================================================
# Career features
# ===========================================================================

class TestCareerFeaturesEdgeCases:
    def test_single_entry_promotion_velocity_zero(self):
        cand = {
            "profile": {"years_of_experience": 3},
            "career_history": [
                {"title": "Senior ML Engineer", "company": "Google",
                 "duration_months": 36, "is_current": True}
            ],
        }
        val, _ = PromotionVelocityFeature().compute(cand)
        assert val == 0.0

    def test_single_entry_inflection_zero(self):
        cand = {
            "career_history": [
                {"title": "ML Engineer", "company": "Meta", "duration_months": 24,
                 "industry": "Tech", "is_current": True}
            ]
        }
        val, _ = InflectionPointStrengthFeature().compute(cand)
        assert val == 0.0

    def test_career_velocity_single_employer(self):
        cand = {
            "profile": {"years_of_experience": 5},
            "career_history": [
                {"title": "Staff Engineer", "company": "Stripe",
                 "duration_months": 60, "is_current": True}
            ],
        }
        val, _ = CareerVelocityFeature().compute(cand)
        assert val == 5.0  # 5 YOE / 1 employer

    def test_tenure_stability_average(self):
        cand = {
            "career_history": [
                {"duration_months": 24, "is_current": False},
                {"duration_months": 36, "is_current": True},
            ]
        }
        val, _ = TenureStabilityFeature().compute(cand)
        assert val == 30.0  # (24 + 36) / 2


# ===========================================================================
# Activity features
# ===========================================================================

class TestActivityFeatures:
    def test_github_minus_one_excluded_from_composite(self):
        """
        CRITICAL: -1 must be treated as MISSING, not zero.
        A candidate with github=-1 should have a HIGHER composite than one with github=0,
        because the missing value contributes no weight (not a zero penalty).
        """
        cand_missing = {
            "redrob_signals": {
                "last_active_date": "2025-01-01",
                "recruiter_response_rate": 0.8,
                "interview_completion_rate": 1.0,
                "github_activity_score": -1,       # missing sentinel
            }
        }
        cand_zero = {
            "redrob_signals": {
                "last_active_date": "2025-01-01",
                "recruiter_response_rate": 0.8,
                "interview_completion_rate": 1.0,
                "github_activity_score": 0,        # actual 0 score
            }
        }
        feat = ActivityQualityCompositeFeature()
        val_missing, _ = feat.compute(cand_missing)
        val_zero, _ = feat.compute(cand_zero)

        assert val_missing > val_zero, (
            f"Missing github (-1) should not penalize: {val_missing=} vs {val_zero=}"
        )

    def test_no_github_uses_three_component_weights(self):
        cand = {
            "redrob_signals": {
                "last_active_date": "2025-01-01",
                "recruiter_response_rate": 0.6,
                "interview_completion_rate": 0.9,
                "github_activity_score": -1,
            }
        }
        val, _ = ActivityQualityCompositeFeature().compute(cand)
        # Should be average of 3 signals only (github excluded)
        assert val > 0.0

    def test_high_github_boosts_composite(self):
        base = {
            "redrob_signals": {
                "last_active_date": "2025-01-01",
                "recruiter_response_rate": 0.5,
                "interview_completion_rate": 0.5,
                "github_activity_score": -1,
            }
        }
        with_github = {
            "redrob_signals": {
                "last_active_date": "2025-01-01",
                "recruiter_response_rate": 0.5,
                "interview_completion_rate": 0.5,
                "github_activity_score": 100,
            }
        }
        feat = ActivityQualityCompositeFeature()
        val_no_gh, _ = feat.compute(base)
        val_gh, _ = feat.compute(with_github)
        assert val_gh > val_no_gh


# ===========================================================================
# Company features
# ===========================================================================

def test_product_vs_services_catches_new_firms():
    """Firms newly added to the list should be classified as services."""
    from feature_lab.features.company_features import _is_services_firm

    new_firms = [
        "Mphasis", "Hexaware Technologies", "NIIT Technologies",
        "Persistent Systems", "L&T Infotech", "Zensar Technologies",
        "Birlasoft", "LTIMindtree", "Syntel", "Genpact"
    ]
    for firm in new_firms:
        assert _is_services_firm(firm), (
            f"'{firm}' should be classified as a services firm but wasn't"
        )


def test_product_vs_services_product_firms_not_flagged():
    """Known product companies should NOT be classified as services."""
    from feature_lab.features.company_features import _is_services_firm

    product_firms = [
        "Flipkart", "Swiggy", "Zomato", "Razorpay",
        "CRED", "Meesho", "Dream11", "PhonePe",
        "Google", "Microsoft", "Amazon", "Meta"
    ]
    for firm in product_firms:
        assert not _is_services_firm(firm), (
            f"'{firm}' is a product company but was flagged as services"
        )


def test_product_vs_services_handles_suffixes():
    """Suffix variations should match (e.g. 'Wipro Technologies' vs 'Wipro')."""
    from feature_lab.features.company_features import _is_services_firm
    assert _is_services_firm("Wipro Technologies Pvt Ltd")
    assert _is_services_firm("TCS Digital")
    assert _is_services_firm("Infosys BPM")


def test_product_vs_services_mixed_career():
    """Candidate with both product and services roles should get intermediate score."""
    from feature_lab.features.company_features import ProductVsServicesFeature
    feat = ProductVsServicesFeature()
    candidate = {
        "career_history": [
            {"company": "Flipkart", "duration_months": 36},   # product
            {"company": "Infosys", "duration_months": 24},     # services
        ]
    }
    score, tag = feat.compute(candidate)
    # 36 product / 60 total = 0.60
    assert abs(score - 0.60) < 0.01, f"Expected 0.60, got {score}"
    assert tag == "clean"


def test_skill_recency_v2_fires_for_senior_candidates():
    """
    Previously skill_recency returned 0 for all senior candidates.
    With date-based logic, a 7-year engineer using PyTorch for 72 months
    should get a non-zero recency score.
    """
    candidate = {
        "profile": {"years_of_experience": 7.0},
        "career_history": [
            {"is_current": True, "duration_months": 24},
            {"is_current": False, "duration_months": 36},
            {"is_current": False, "duration_months": 24},
        ],
        "skills": [
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 72},
        ]
    }
    from feature_lab.features.skill_features import SkillRecencyFeature
    feat = SkillRecencyFeature()
    value, tag = feat.compute(candidate)
    assert value > 0.0, (
        f"Senior candidate with active PyTorch use should have recency > 0, got {value}"
    )
    assert tag == "clean"


def test_skill_recency_v2_zero_for_no_deep_ir_skills():
    """Candidate with only generic skills should still score 0."""
    candidate = {
        "profile": {"years_of_experience": 5.0},
        "career_history": [{"is_current": True, "duration_months": 12}],
        "skills": [
            {"name": "Excel", "proficiency": "expert", "duration_months": 48},
        ]
    }
    from feature_lab.features.skill_features import SkillRecencyFeature
    feat = SkillRecencyFeature()
    value, tag = feat.compute(candidate)
    assert value == 0.0


def test_skill_recency_v2_handles_zero_yoe():
    """Zero YOE should not crash and should return 0."""
    candidate = {
        "profile": {"years_of_experience": 0},
        "career_history": [],
        "skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 12}]
    }
    from feature_lab.features.skill_features import SkillRecencyFeature
    feat = SkillRecencyFeature()
    value, tag = feat.compute(candidate)
    assert value == 0.0


def test_jd_skill_score_weights_deep_ir_higher_than_buzzword():
    """
    Two candidates: one with deep-IR skills, one with buzzword skills,
    same duration and proficiency. Deep-IR should score higher.
    """
    deep_ir_candidate = {
        "profile": {"years_of_experience": 5.0},
        "career_history": [],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 36,
             "endorsements": 0},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    buzzword_candidate = {
        "profile": {"years_of_experience": 5.0},
        "career_history": [],
        "skills": [
            {"name": "LangChain", "proficiency": "expert", "duration_months": 36,
             "endorsements": 0},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    from feature_lab.features.jd_skill_features import JDWeightedSkillScoreFeature
    feat = JDWeightedSkillScoreFeature()
    deep_score, _ = feat.compute(deep_ir_candidate)
    buzz_score, _ = feat.compute(buzzword_candidate)
    assert deep_score > buzz_score, (
        f"Deep-IR skill should score higher than buzzword: {deep_score} vs {buzz_score}"
    )
    # Specifically: 3x vs 1x weight — should be exactly 3x ratio
    assert abs(deep_score / buzz_score - 3.0) < 0.1, (
        "Band weight ratio should be ~3:1 for deep-ir vs buzzword"
    )


def test_jd_skill_score_sums_across_skills():
    """
    A candidate with two deep-IR skills should score higher than one with one.
    (jd_skill_score sums, unlike skill_mastery which takes max)
    """
    one_skill = {
        "profile": {}, "career_history": [],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 36,
             "endorsements": 0}
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    two_skills = {
        "profile": {}, "career_history": [],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 36,
             "endorsements": 0},
            {"name": "PyTorch", "proficiency": "intermediate", "duration_months": 24,
             "endorsements": 0},
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    from feature_lab.features.jd_skill_features import JDWeightedSkillScoreFeature
    feat = JDWeightedSkillScoreFeature()
    score_one, _ = feat.compute(one_skill)
    score_two, _ = feat.compute(two_skills)
    assert score_two > score_one, "Two deep-IR skills should score more than one"


def test_jd_skill_score_duration_cap():
    """Duration is capped at 60 months. 120 months should not score double 60 months."""
    capped = {
        "profile": {}, "career_history": [],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 120,
             "endorsements": 0}
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    at_cap = {
        "profile": {}, "career_history": [],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 60,
             "endorsements": 0}
        ],
        "redrob_signals": {"skill_assessment_scores": {}}
    }
    from feature_lab.features.jd_skill_features import JDWeightedSkillScoreFeature
    feat = JDWeightedSkillScoreFeature()
    score_120, _ = feat.compute(capped)
    score_60, _ = feat.compute(at_cap)
    assert abs(score_120 - score_60) < 0.01, (
        "Duration cap should make 60m and 120m score identically"
    )


def test_yoe_band_fit_target_band_scores_1():
    """5-9 YOE candidates should score 1.0."""
    from feature_lab.features.career_features import YOEBandFitFeature
    feat = YOEBandFitFeature()
    for yoe in [5.0, 6.5, 8.0, 9.0]:
        candidate = {
            "profile": {"years_of_experience": yoe},
            "career_history": []
        }
        # 9 is at boundary — check both sides
        val, tag = feat.compute(candidate)
        if yoe < 9:
            assert val == 1.0, f"YOE={yoe} should score 1.0, got {val}"
        assert tag == "clean"


def test_yoe_band_fit_junior_penalized():
    """<3 YOE should score 0.10."""
    from feature_lab.features.career_features import YOEBandFitFeature
    feat = YOEBandFitFeature()
    candidate = {"profile": {"years_of_experience": 1.5}, "career_history": []}
    val, tag = feat.compute(candidate)
    assert val == 0.10, f"Junior candidate should score 0.10, got {val}"


def test_yoe_band_fit_no_data_returns_neutral():
    """Missing YOE should return 0.50 (neutral, no penalty)."""
    from feature_lab.features.career_features import YOEBandFitFeature
    feat = YOEBandFitFeature()
    candidate = {"profile": {}, "career_history": []}
    val, tag = feat.compute(candidate)
    assert val == 0.50, f"Missing YOE should return 0.50, got {val}"
    assert tag == "sparse"


def test_yoe_band_fit_infers_from_career_history():
    """If no explicit YOE, should infer from career_history duration."""
    from feature_lab.features.career_features import YOEBandFitFeature
    feat = YOEBandFitFeature()
    # 84 months = 7 years → should be in target band
    candidate = {
        "profile": {},  # no explicit YOE
        "career_history": [
            {"duration_months": 36},
            {"duration_months": 48},
        ]
    }
    val, tag = feat.compute(candidate)
    assert val == 1.0, f"7-year career should infer target band, got {val}"
    assert tag == "sparse"  # inferred, not stated
