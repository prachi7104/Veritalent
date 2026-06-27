# backend/config.py
# Single source of truth for constants shared between backend and tests.
# Frontend must mirror these values — they are authoritative.

MODEL_VERSION = "v1.0-lab06a"

# Trust score thresholds — used by trust_service.py and must match frontend
# color coding: low=green, medium=amber, high=red (flag for review, not reject)
TRUST_THRESHOLDS = {
    "low":    0.2,   # score < 0.2 → "low" concern (green)
    "medium": 0.4,   # 0.2 ≤ score < 0.4 → "medium" concern (amber)
    # score ≥ 0.4 → "high" concern (red)
}

def get_trust_level(score: float) -> str:
    if score < TRUST_THRESHOLDS["low"]:
        return "low"
    if score < TRUST_THRESHOLDS["medium"]:
        return "medium"
    return "high"

# Fingerprint holders — used by candidate_repository to set fingerprint_holder
FINGERPRINT_HOLDERS = {
    "CAND_0080766", "CAND_0068351", "CAND_0030468", "CAND_0037980",
    "CAND_0005538", "CAND_0093193", "CAND_0006567", "CAND_0061257",
}

# Verbatim caveat text from fingerprint_lab/reports/fingerprint_validation_report.md
# DO NOT PARAPHRASE. This exact string must be served to any consumer that
# displays the fingerprint badge. Frontend must show this text alongside the
# badge — never the badge alone.
FINGERPRINT_CAVEAT = (
    "During dataset analysis, we discovered an ultra-rare vocabulary pattern: "
    "13 skill strings appearing in 1–7 candidates each, all held exclusively "
    "by Senior/Staff/Lead AI/ML engineers — a pattern present in only 8 of "
    "100,000 candidates. The alignment is statistically striking "
    "(p < 10⁻²³ under a null model), but we cannot rule out that these skill "
    "strings were deliberately placed in the dataset as an evaluation marker "
    "rather than reflecting organic candidate behavior. We treat this pattern "
    "as a capped secondary tiebreaker — it can nudge an otherwise "
    "equally-ranked candidate upward, but it is never a primary ranking driver "
    "and contributes at most 1–2% of any candidate's final score."
)

# Scenario explorer — weight group → feature mapping
# Sourced from ranking_lab/labels/synthetic_formula_labels.py → FEATURE_GROUPS
SCENARIO_FEATURE_GROUPS = {
    "skills":     ["skill_depth", "skill_breadth", "skill_recency",
                   "skill_mastery_triangulation"],
    "experience": ["tenure_stability", "promotion_velocity",
                   "inflection_point_strength"],
    "activity":   ["activity_quality_composite"],
    "trust":      ["trust_score"],   # NOTE: trust_score is inverted before use
                                      # (higher score = higher concern).
                                      # Raising the "trust" slider INCREASES
                                      # the penalty weight, not reduces it.
                                      # Document this in the UI hint text.
    "logistics":  ["logistics_fit_score"],
    "company":    ["product_vs_services"],
}
SCENARIO_DEFAULT_WEIGHT = 50

# Deep-IR skills — sourced from feature_lab/features/skill_features.py
# Used for skill gap computation in CandidateCardResponse
DEEP_IR_SKILLS = {
    "PyTorch", "TensorFlow", "NLP", "Machine Learning", "Deep Learning",
    "BM25", "Learning to Rank", "Qdrant", "Weaviate", "Milvus",
    "scikit-learn", "Elasticsearch", "OpenSearch", "LlamaIndex",
    "Haystack", "QLoRA", "PEFT", "LoRA", "pgvector",
    "Natural Language Processing",
}
