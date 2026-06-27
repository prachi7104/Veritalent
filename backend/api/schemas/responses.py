from pydantic import BaseModel, Field
from typing import Optional, Any

class JDDecomposition(BaseModel):
    must_haves: list[str]
    nice_to_haves: list[str]
    hard_exclusions: list[str]
    experience_band: str
    logistics: dict[str, Any]

class FeatureContribution(BaseModel):
    feature_name: str
    value: float
    shap_contribution: float
    direction: str  # "positive" | "negative"

class SkillGap(BaseModel):
    missing_deep_ir_skills: list[str]
    matched_deep_ir_skills: list[str]
    gap_to_next_tier: Optional[str] = None

class FunnelStats(BaseModel):
    total_pool: int
    title_relevant: int
    retrieved: int
    shown: int

class CandidateCardResponse(BaseModel):
    candidate_id: str
    rank: int
    score: float
    current_title: str
    current_company: str
    years_of_experience: float
    location: str
    top_features: list[FeatureContribution]
    trust_score: float
    trust_level: str  # "low" | "medium" | "high"
    fingerprint_holder: bool
    narrative: str
    narrative_is_llm: bool
    fallback_used: bool
    skill_gap: SkillGap

class SearchResponse(BaseModel):
    session_id: str
    funnel_stats: FunnelStats
    candidates: list[CandidateCardResponse]
    jd_decomposition: JDDecomposition

class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int
    is_current: bool
    industry: str

class ProfileDetail(BaseModel):
    current_title: str
    current_company: str
    years_of_experience: float
    headline: str
    summary: str
    location: str
    country: str
    career_history: list[CareerEntry]

class CandidateFeatures(BaseModel):
    skill_depth: float
    skill_breadth: float
    skill_mastery_triangulation: float
    skill_recency: float
    implied_skill_score: float
    career_velocity: float
    tenure_stability: float
    promotion_velocity: float
    activity_quality_composite: float
    industry_relevance: float
    logistics_fit_score: float
    product_vs_services: float
    trust_score: float

class TrustCheck(BaseModel):
    score: float
    flagged: bool
    explanation: str

class YoeTenureConsistencyCheck(TrustCheck):
    deviation_years: float

class ProficiencyPlausibilityCheck(TrustCheck):
    implausible_skill_count: int
    severity_weighted_sum: float

class KeywordStuffingDensityCheck(TrustCheck):
    total_claimed_skills: int
    years_experience: float
    density_ratio: float

class AssessmentCorroborationCheck(TrustCheck):
    coverage: float
    has_data: bool

class TemplateRelianceCheck(TrustCheck):
    reliance_fraction: float
    note: str

class TrustChecks(BaseModel):
    yoe_tenure_consistency: YoeTenureConsistencyCheck
    proficiency_plausibility: ProficiencyPlausibilityCheck
    keyword_stuffing_density: KeywordStuffingDensityCheck
    assessment_corroboration: AssessmentCorroborationCheck
    template_reliance: TemplateRelianceCheck

class TrustBreakdown(BaseModel):
    composite_score: float
    level: str  # "low" | "medium" | "high"
    checks: TrustChecks
    caveat: str

class SHAPAttribution(BaseModel):
    top_features: list[FeatureContribution]
    baseline_score: float

class CandidateDetailResponse(BaseModel):
    candidate_id: str
    profile: ProfileDetail
    features: CandidateFeatures
    trust_breakdown: TrustBreakdown
    shap_attribution: SHAPAttribution
    narrative: str
    narrative_is_llm: bool
    fingerprint_holder: bool

class ComparisonMatrix(BaseModel):
    features: list[str]
    values: dict[str, dict[str, float]]
    deltas: dict[str, dict[str, float]]

class CompareResponse(BaseModel):
    candidates: list[CandidateDetailResponse]
    comparison_matrix: ComparisonMatrix

class ScenarioCandidate(BaseModel):
    candidate_id: str
    new_rank: int
    original_rank: int
    rank_delta: int
    scenario_score: float

class ScenarioRerankResponse(BaseModel):
    re_ranked: list[ScenarioCandidate]
    weight_applied: dict[str, float]
    note: str

class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    model_version: str
    feature_store_rows: int
    feature_store_freshness: str
    narratives_cached: int
    dense_index_loaded: bool
    retrieval_model: str
