# Backend Architecture Specification (v2)

## Overview
This document specifies the production architecture for the Veritalent Candidate Discovery Engine.

## Components
1. **Data Access Layer**
   - `candidate_repository.py`: Loads `candidates.jsonl` into memory as a dictionary.
   - `feature_store_repository.py`: Loads `feature_store.jsonl` into memory.

2. **Services Layer**
   - `jd_decomposition_service.py`: Uses `llama-3.3-70b-versatile` or `gpt-oss-120b` for JD decomposition, with a keyword-extraction fallback. Results are cached.
   - `retrieval_service.py`: Wraps `DenseIndex` over `BAAI/bge-small-en-v1.5`, using exact allowlist candidates.
   - `feature_service.py`: Fast memory lookup for candidate features.
   - `ranking_service.py`: Loads LightGBM Lambdarank booster. Strict ordering checks against `TRAINING_FEATURES`.
   - `scenario_service.py`: Linear baseline re-ranking based on weight sliders (Skills, Experience, Activity, Trust, Logistics, Company).
   - `explainability_service.py`: Loads SHAP TreeExplainer and integrates the LLM narrative fallback pipeline.
   - `skill_gap_helper.py`: Computes missing/matched Deep IR skills and actionable next steps.
   - `trust_service.py`: Translates raw trust signals into the `TrustBreakdown` API schema.

3. **Pipelines Layer**
   - `live_query_pipeline.py`: Orchestrates search requests across all services within the 800ms time budget.

4. **API Layer (FastAPI)**
   - `/search`: Primary entry point for semantic search and ranking.
   - `/rerank`: Re-evaluates ranking using cached representations without external API calls.
   - `/candidate/{candidate_id}`: Returns detailed profile, SHAP attributions, trust breakdown, and explanations.
   - `/scenarios/rerank`: Reranks an active session dynamically using linear weight sliders.
   - `/compare`: Compares 2-4 candidates side-by-side using key features.
   - `/health`: Exposes system status and resource readiness.
