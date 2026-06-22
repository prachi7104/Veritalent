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
from feature_lab.features.company_features import ProductVsServicesClassificationFeature


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
# Company features — sequence-dependent test (required by task spec)
# ===========================================================================

class TestProductVsServicesSequence:
    def test_career_long_product_scores_one(self):
        cand = {
            "career_history": [
                {"company": "Google", "is_current": True},
                {"company": "Meta", "is_current": False},
            ]
        }
        val, _ = ProductVsServicesClassificationFeature().compute(cand)
        assert val == 1.0

    def test_career_long_services_scores_zero(self):
        cand = {
            "career_history": [
                {"company": "Infosys", "is_current": True},
                {"company": "Wipro", "is_current": False},
            ]
        }
        val, _ = ProductVsServicesClassificationFeature().compute(cand)
        assert val == 0.0

    def test_currently_services_prior_product_scores_zero_point_eight(self):
        """
        KEY TEST: A candidate who moved TO services AFTER years at a product company
        must NOT be penalised the same as a career-long services candidate.
        Expected score: 0.8 (JD carve-out).
        """
        cand = {
            "career_history": [
                {"company": "TCS", "is_current": True},         # now at services
                {"company": "Razorpay", "is_current": False},   # prior product
                {"company": "Swiggy", "is_current": False},     # prior product
            ]
        }
        val, _ = ProductVsServicesClassificationFeature().compute(cand)
        assert val == 0.8, f"Expected 0.8 (carve-out), got {val}"

    def test_currently_product_prior_services_scores_zero_point_nine(self):
        cand = {
            "career_history": [
                {"company": "Stripe", "is_current": True},      # now at product
                {"company": "Cognizant", "is_current": False},  # prior services
            ]
        }
        val, _ = ProductVsServicesClassificationFeature().compute(cand)
        assert val == 0.9

    def test_services_and_product_classification_are_distinct(self):
        """Prove 0.8 ≠ 0.0 — the two scenarios must not conflate."""
        career_long_services = {"career_history": [{"company": "TCS", "is_current": True}]}
        moved_to_services = {
            "career_history": [
                {"company": "TCS", "is_current": True},
                {"company": "Google", "is_current": False},
            ]
        }
        feat = ProductVsServicesClassificationFeature()
        score_ls, _ = feat.compute(career_long_services)
        score_ms, _ = feat.compute(moved_to_services)
        assert score_ms > score_ls
        assert score_ls == 0.0
        assert score_ms == 0.8
