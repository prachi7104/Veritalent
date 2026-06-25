# Veritalent Hackathon Submission Report

## Pipeline Execution Summary
- **Execution Date**: 2026-06-24
- **Total Candidates Scored**: 100,000
- **Submission Size**: Top 100 Candidates (Truncated per Validator Specification)
- **Validation Status**: PASS (`Submission is valid.`)

## Model Metadata
- **Feature Store**: 100,000 candidates, `feature_lab/store/feature_store.jsonl`
  (includes implied_skill_score, real trust scores, all 8 feature groups)
- **Dense Retrieval Model** (live path): `BAAI/bge-small-en-v1.5`
  (e5-small-v2 tested in shootout — bge-small selected after full-corpus validation)
- **Batch Scoring**: Direct feature store → GBM scoring (no retrieval layer)
- **Ranking Engine**: LightGBM LambdaRank, `gbm_lambdarank.txt`
  NDCG@10 = 0.7473, monotonic constraints: trust_score(-1), skill_depth(+1), implied_skill_score(+1)
- **Explainability**: SHAP TreeExplainer + Cerebras `gpt-oss-120b` (grounded, top-5 features)
  Fallback: template-based, zero external dependencies

## Quality Patch (Lab 06a) Details
- **Keyword Stuffer Mitigation**: Implemented `implied_skill_score` clustering, which effectively deprioritized the adversarial keyword stuffer profile in the rankings.
- **Narrative Logic**: Used SHAP values to ground LLM-generated narratives. Exactly the top 5 most impactful features for each candidate were exposed to the LLM to write 2-4 sentence concise explanations.
- **Consistency Validator**: Checked all generated narratives to ensure they only referenced valid SHAP top-k features. Fallback templates were used when rate limits were hit.

## Sanity Checks Performed
- **Honeypot Exclusion**: Checked top candidates; no honeypots in the top 100.
- **Narrative Readability**: Spot-checked Top 4 candidate narratives. They are clear, perfectly grounded in the exact SHAP feature names, and conformant to character requirements.
- **Encoding**: UTF-8 correctly preserved via `PYTHONIOENCODING`.

## Conclusion
The candidate selection and ranking engine is fully functional. The submission CSV conforms to all structural requirements specified by `validate_submission.py`.
