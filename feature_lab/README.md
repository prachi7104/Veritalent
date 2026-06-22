# Feature Engineering Lab

Reproducible feature store and ablation testing for the Redrob Hackathon
Intelligent Candidate Discovery & Ranking Engine.

---

## Directory Structure

```
feature_lab/
├── features/
│   ├── base.py                  # BaseFeature ABC: name, version, reliability_tag
│   ├── feature_registry.py      # Global registry singleton
│   ├── skill_features.py        # skill_depth, skill_breadth, skill_recency,
│   │                            # skill_mastery_triangulation
│   ├── career_features.py       # career_velocity, promotion_velocity,
│   │                            # tenure_stability, inflection_point_strength
│   ├── trust_features.py        # trust_score_composite (stub — see note)
│   ├── activity_features.py     # activity_quality_composite
│   ├── industry_features.py     # industry_relevance
│   ├── logistics_features.py    # logistics_fit_score
│   └── company_features.py      # product_vs_services
├── store/
│   ├── schema.py                # Explicit column schema + validate_schema()
│   └── feature_store.py        # Streams candidates.jsonl → feature_store.jsonl
├── ablation/
│   ├── ablation_runner.py       # Linear weighted scorer, NDCG@50 ablation
│   └── correlation_check.py    # Pairwise correlation audit (threshold 0.7)
├── reports/
│   └── feature_ablation_report.md   # Full ablation results (real NDCG values)
└── tests/
    ├── conftest.py              # sys.path setup for pytest
    └── test_features.py         # 20 unit tests covering all edge cases
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install tqdm pandas numpy
```

### 2. Generate the feature store (full 100k pool)

```bash
python -m feature_lab.store.feature_store \
    --input  "dataset/[PUB] India_runs_data_and_ai_challenge/.../candidates.jsonl" \
    --output "feature_lab/store/feature_store.jsonl"
```

Runtime: ~12 seconds for 100,000 candidates on a modern laptop.

### 3. Run ablation testing

```bash
python -m feature_lab.ablation.ablation_runner \
    --store feature_lab/store/feature_store.jsonl
```

Gold set path is configured via a single constant in `ablation_runner.py`:
```python
GOLD_SET_POOLED_PATH = r"...\retrieval_lab\evaluation\gold_set_pooled.json"
GOLD_SET_FALLBACK_PATH = r"...\retrieval_lab\evaluation\gold_set.json"
```
Swap the active path with a one-line change. No grep-and-replace needed.

### 4. Run correlation check

```bash
python -m feature_lab.ablation.correlation_check \
    --store feature_lab/store/feature_store.jsonl
```

### 5. Run unit tests

```bash
pytest feature_lab/tests/test_features.py -v
```

Expected: **20 passed** in < 1 second.

---

## Feature Summary

| Feature | Group | Reliability Tag | Notes |
|---------|-------|----------------|-------|
| `skill_depth` | skill | clean | Max deep-IR duration × proficiency weight |
| `skill_breadth` | skill | clean | Band coverage fraction (4 bands) |
| `skill_recency` | skill | clean | Skill extends into current role (binary 0/1) |
| `skill_mastery_triangulation` | skill | clean/sparse | clean only with assessment corroboration |
| `career_velocity` | career | clean | YOE / employers (inverted, stability proxy) |
| `promotion_velocity` | career | clean | Title rank delta / YOE |
| `tenure_stability` | career | clean | Avg duration_months per role |
| `inflection_point_strength` | career | experimental | Largest role-transition jump |
| `trust_score` | trust | sparse | **Stub** — replace with Lab 04 output |
| `activity_quality_composite` | activity | clean | recency + RRR + ICR + GitHub (missing-safe) |
| `industry_relevance` | industry | clean | Max industry tier across career history |
| `logistics_fit_score` | logistics | clean | Notice decay + mode + location |
| `product_vs_services` | company | clean | Sequence-aware (full career_history scan) |

---

## Critical Constraints Enforced

| Constraint | Where |
|-----------|-------|
| `github_activity_score == -1` → missing, not zero | `activity_features.py` L63–65 |
| `education.start_year`/`end_year` → never used | Not referenced anywhere |
| Trust score → continuous float, never binary gate | `trust_features.py` stub |
| Title-tier → soft prior only, no hard gate | Not used in any filter logic |
| `product_vs_services` → full career_history, not just current_company | `company_features.py` loop |

---

## Ablation Summary (see full report in `reports/feature_ablation_report.md`)

| Group | Δ NDCG@50 | Verdict |
|-------|-----------|---------|
| skill | −0.3882 | **Must Keep** |
| career | −0.0871 | **Must Keep** |
| activity | −0.0451 | **Must Keep** |
| logistics | −0.0237 | Keep, de-weight candidate |
| industry | +0.0168 | Low marginal value — keep for explainability |
| trust (stub) | 0.0000 | Low marginal value — re-evaluate after Lab 04 |
| company | 0.0000 | Low marginal value — keep for explainability |

---

## Design Decisions

- **Ablation uses linear scoring, not LightGBM.** Training a GBM on ~80 gold examples
  per CV fold produces noise. Linear scoring is deterministic and interpretable.
  GBM training belongs in Lab 06.

- **Registry is a module-level singleton.** All feature modules self-register at
  import time. The `feature_lab/__init__.py` eagerly imports all modules, so any
  code that does `import feature_lab` will have a fully-populated registry.

- **`trust_features.py` is a stub** with the correct output shape:
  `{"trust_score": 0.5, "trust_reasons": [], "reliability_tag": "sparse"}`.
  Dropping in the real Lab 04 module requires no schema changes.
