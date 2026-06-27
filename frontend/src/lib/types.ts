export interface SearchRequest {
  jd_text: string;
  top_k?: number;
  include_trust?: boolean;
}

export interface FunnelStats {
  total_pool: number;
  title_relevant: number;
  retrieved: number;
  shown: number;
}

export interface JDDecomposition {
  must_haves: string[];
  nice_to_haves: string[];
  hard_exclusions: string[];
  experience_band: string;
  logistics: Record<string, unknown>;
  fallback_used?: boolean;
}

export interface FeatureContribution {
  feature_name: string;
  value: number;
  shap_contribution: number;
  direction: "positive" | "negative";
}

export interface SkillGap {
  missing_deep_ir_skills: string[];
  matched_deep_ir_skills: string[];
  gap_to_next_tier: string | null;
}

export type TrustLevel = "low" | "medium" | "high";

// No candidate name field exists anywhere in this schema. Render by
// current_title / current_company / location, not by name.
export interface CandidateCardResponse {
  candidate_id: string;
  rank: number;
  score: number; // raw LambdaRank score, can be negative — never render as a %
  current_title: string;
  current_company: string;
  years_of_experience: number;
  location: string;
  top_features: FeatureContribution[];
  trust_score: number;
  trust_level: TrustLevel;
  fingerprint_holder: boolean;
  narrative: string;
  narrative_is_llm: boolean;
  fallback_used: boolean;
  skill_gap: SkillGap;
}

export interface SearchResponse {
  session_id: string;
  funnel_stats: FunnelStats;
  candidates: CandidateCardResponse[];
  jd_decomposition: JDDecomposition;
  warnings?: string[];
}

export interface CareerEntry {
  company: string;
  title: string;
  start_date: string;
  end_date: string | null;
  duration_months: number;
  is_current: boolean;
  industry: string;
}

export interface TrustCheck {
  score: number;
  flagged: boolean;
  explanation: string;
}

export interface TrustBreakdown {
  composite_score: number;
  level: TrustLevel;
  checks: {
    yoe_tenure_consistency: TrustCheck & { deviation_years: number };
    proficiency_plausibility: TrustCheck & {
      implausible_skill_count: number;
      severity_weighted_sum: number;
    };
    keyword_stuffing_density: TrustCheck & {
      total_claimed_skills: number;
      years_experience: number;
      density_ratio: number;
    };
    assessment_corroboration: TrustCheck & { coverage: number; has_data: boolean };
    template_reliance: TrustCheck & { reliance_fraction: number; note: string };
  };
  caveat: string; // always render, never optional
}

export interface SHAPAttribution {
  top_features: FeatureContribution[];
  baseline_score: number;
}

// Deliberately excludes rank, score, and skill_gap — those only exist on
// CandidateCardResponse from /search. A detail view needs them passed in
// from whichever card the user clicked, not re-fetched.
export interface CandidateDetailResponse {
  candidate_id: string;
  profile: {
    current_title: string;
    current_company: string;
    years_of_experience: number;
    headline: string;
    summary: string;
    location: string;
    country: string;
    career_history: CareerEntry[];
  };
  features: Record<string, number>;
  trust_breakdown: TrustBreakdown;
  shap_attribution: SHAPAttribution;
  narrative: string;
  narrative_is_llm: boolean;
  fingerprint_holder: boolean;
}

export interface CompareRequest {
  candidate_ids: string[]; // 2-4
}

export interface CompareResponse {
  candidates: CandidateDetailResponse[];
  comparison_matrix: {
    features: string[];
    values: Record<string, Record<string, number>>;
    deltas: Record<string, Record<string, number>>; // signed, vs. top-ranked
  };
}

export interface RerankRequest {
  session_id: string;
  updated_jd_text: string;
}

export type ScenarioGroup = "skills" | "experience" | "activity" | "trust" | "logistics" | "company";

export interface ScenarioRerankRequest {
  session_id: string;
  weight_overrides: Partial<Record<ScenarioGroup, number>>;
}

export interface ScenarioCandidate {
  candidate_id: string;
  new_rank: number;
  original_rank: number;
  rank_delta: number;
  scenario_score: number;
}

export interface ScenarioRerankResponse {
  re_ranked: ScenarioCandidate[];
  weight_applied: Record<ScenarioGroup, number>;
  note: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  model_version: string;
  feature_store_rows: number;
  feature_store_freshness: string;
  narratives_cached: number;
  dense_index_loaded: boolean;
  retrieval_model: string;
}
