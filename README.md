# Candidate Discovery & Ranking Platform

An AI-powered candidate discovery and ranking system built for a
recruiting-focused hackathon. It searches a pool of 100,000 candidate
profiles against a job description, ranks them with an explainable model,
flags trust concerns without auto-rejecting anyone, and lets a recruiter
reshape the ranking themselves instead of just trusting a black box.

---

## Submission

| Item | Details |
|------|---------|
| **Ranked output** | `submission/submission_v2.csv` — 100 candidates, passes `validate_submission.py` |
| **Ranking model** | LightGBM LambdaRank · NDCG@10 = **0.7473** (GBM) → **0.7482** (BlendScorer) |
| **Scoring formula** | `0.90 × norm(GBM_score) + 0.10 × norm(jd_skill_score)` (yoe_band_fit removed after ablation proved zero marginal contribution) |
| **Explainability** | SHAP TreeExplainer + Cerebras `gpt-oss-120b` grounded narratives · 0% hallucination (Lab 07) |
| **Adversarial robustness** | Keyword stuffer: −3.26 vs. legit: −0.32 · Consistent fraud & activity faker: all PASS |
| **Retrieval** | 7-experiment shootout · `bge-small-en-v1.5` dense index (live path, ~78ms) |
| **Validation** | `Submission is valid.` |

For full ablation tests, validation details, and dataset statistics, see the [EVALUATION_SUMMARY.md](file:///c:/projects/Veritalent/EVALUATION_SUMMARY.md).

To regenerate the submission CSV from scratch:

```bash
python submission/batch_scoring_pipeline.py --output submission/submission_v2.csv --top-n 100
python dataset/"[PUB] India_runs_data_and_ai_challenge"/India_runs_data_and_ai_challenge/validate_submission.py \
  --submission submission/submission_v2.csv
```

---

## Lab results summary

| Lab | Key finding |
|-----|-------------|
| Retrieval Lab | Dense-only (`bge-small-en-v1.5`) wins live path at ~78ms; Skill Graph Recall best offline (NDCG@50=0.4528); pooling-bias fix revealed 65 valid candidates invisible to BM25 |
| Ranking Lab | LambdaRank (+0.1899 NDCG@10 vs. linear baseline); cross-encoder ensemble tested but adds ~11s latency — excluded from live path |
| Feature Lab | `skill_mastery_triangulation` dominates (split gain 209); `industry_relevance` excluded (ablation: +0.0168 NDCG without it); `career_velocity` display-only (r=0.75 with `tenure_stability`) |
| Trust Lab | ~70/~80 documented honeypots caught; keyword stuffer PASS on current model; sophisticated internally-consistent fraud is a disclosed limitation |
| Explainability Lab | 0% hallucination rate on LLM-grounded narratives; SHAP coverage 100%; fallback templates factually verified |

---

## Table of contents

- [The problem](#the-problem)
- [What we built](#what-we-built)
- [Why it's different](#why-its-different)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Screens](#screens)
- [API reference](#api-reference)
- [Trust & transparency principles](#trust--transparency-principles)
- [Setup](#setup)
- [Running tests](#running-tests)

---

## The problem

Recruiters sourcing for a role get handed a pile of hundreds or thousands of
profiles and a job description, and have to figure out — fast — who's
actually worth a closer look. Most tools that try to automate this either:

- act as a black box that spits out a ranked list with no explanation, or
- bolt on "AI scoring" that nobody on the hiring team trusts enough to act on.

We built a system that does both the ranking *and* the showing-its-work, with
trust signals that are explicit about what they do and don't catch, and a way
for a recruiter to push back on the model's assumptions instead of just
accepting them.

## What we built

A two-sided system: a backend that retrieves, ranks, and explains candidates
against a job description, and a frontend that makes every part of that
pipeline visible and adjustable.

- **Search** — paste or describe a role, get a ranked shortlist in under a
  second.
- **Explainability** — every candidate's rank comes with the actual SHAP
  feature contributions that produced it, not a generic summary.
- **Trust auditing** — a five-check trust breakdown per candidate (tenure
  consistency, proficiency plausibility, keyword stuffing density, assessment
  corroboration, template reliance), each with its own explanation, plus a
  standing disclaimer about what the system can't catch.
- **Scenario exploration** — six weight sliders let a recruiter re-rank the
  same shortlist under a different model (linear, not the primary GBM) to see
  how sensitive the ranking is to their own priorities.
- **Skill gap counterfactuals** — for anyone outside the top 3, a plain-language
  estimate of what would move them closer to it.
- **Shortlisting and comparison** — pull candidates out of the ranked list
  and compare them side by side against the model's own features.

## Why it's different

A few decisions we made on purpose, because they came up as real failure
modes while building this rather than as nice-to-haves:

- **We never show a fabricated match percentage.** The ranking model's raw
  score is an unbounded LambdaRank value — it can be negative. The only
  number we surface as a headline is rank, because it's the one thing the
  model actually guarantees.
- **We disclose when rankings are statistically tied.** Candidates ranked
  7–12 in a typical search score within 0.000082 of each other — that's
  tie-breaking by candidate ID, not a meaningful quality signal, and the UI
  says so instead of implying false precision.
- **The trust score says what it can't do, every time it's shown.** "This
  trust score detects sloppy or inconsistent fraud only. It does not
  reliably catch sophisticated, internally-consistent fraud" is not a
  buried disclaimer — it renders next to the score every time.
- **An anomalous statistical pattern gets disclosed, not exploited.**
  During data analysis we found an ultra-rare vocabulary pattern held by
  exactly 8 of 100,000 candidates, all senior AI/ML engineers, with a
  p-value under 10⁻²³. We could not rule out that it was a planted
  evaluation marker rather than organic signal — so it's excluded from the
  ranking model entirely and capped as a secondary tiebreaker worth at most
  1–2% of any score, and the full caveat is shown verbatim next to the badge
  any time it appears. We'd rather under-use a suspicious signal and say so
  than quietly benefit from it.
- **Comparison deltas are never color-coded as good/bad.** "Higher is
  better" doesn't hold for every feature — trust_score is explicitly
  inverted — so a red/green delta would misrepresent roughly half the rows.

## Architecture

```
                              ┌─────────────────────────┐
                              │     Frontend (Next.js)   │
                              │  Home → Discovery →      │
                              │  Detail / Compare /      │
                              │  Shortlist / Demo Mode    │
                              └────────────┬─────────────┘
                                           │ REST (JSON)
                              ┌────────────▼─────────────┐
                              │      FastAPI backend      │
                              └────────────┬─────────────┘
        ┌──────────────┬────────────────────┼────────────────┬───────────────┐
        ▼              ▼                    ▼                ▼               ▼
  JD decomposition  Retrieval          Feature lookup    Ranking        Explainability
  (LLM + SHA-256     (bge-small-en-     (in-memory dict   (GBM           (SHAP + cached
  cache, keyword     v1.5 dense index,  from feature_     LambdaRank,    narratives,
  fallback)          31,179-candidate   store.jsonl,      monotonic on   fallback
                     allowlist funnel)  100k candidates)  trust/skill)   template if
                                                                          no cache hit)
                                           │
                                           ▼
                                  Scenario service (TTLCache
                                  session store, linear baseline,
                                  6 weight groups, never the GBM)
```

### Request flow — live search

`POST /search` → JD decomposition (cache hit, or LLM with an 8s timeout
falling back to keyword extraction) → dense retrieval over the allowlist
funnel → feature store lookup → GBM scoring → session created for the
scenario explorer → per-candidate SHAP + narrative + skill gap assembled →
`SearchResponse` returned.

### Request flow — scenario exploration

Slider change → `POST /scenarios/rerank` → session lookup (404 if expired,
TTL 3600s) → pure arithmetic linear re-rank over already-loaded features, no
embedding inference, no LLM, no GBM → re-ranked list with `rank_delta`
against the original GBM order.

## Tech stack

**Backend:** FastAPI, LightGBM (LambdaRank + BlendScorer post-processing), SHAP
(TreeExplainer for feature attribution), `sentence-transformers` (bge-small-en-v1.5
dense index), `cachetools` (TTLCache for sessions and SHA-256-keyed JD decomposition
cache), Cerebras `gpt-oss-120b` / Groq for JD decomposition and SHAP-grounded
narrative generation — both with a fully offline keyword-extraction / template fallback.

**Scoring formula (batch submission path):**
`final_score = 0.90 × norm(GBM_score) + 0.10 × norm(jd_skill_score)` — yoe_band_fit
removed after ablation confirmed zero marginal NDCG contribution.

**Frontend:** Next.js (App Router), TypeScript, vanilla CSS (no Tailwind —
styling runs entirely through a small set of CSS custom properties in
`tokens.css`), `lucide-react` for icons, a React Context provider
(`SearchStateProvider`) as the single client-side data layer instead of prop
drilling or URL-encoded state.

## Repository structure

```
backend/
├── main.py                  # FastAPI entrypoint
├── config.py                # TRUST_THRESHOLDS, FINGERPRINT_CAVEAT, etc. — single source of truth
├── api/routes/               # search, candidate, compare, scenarios, health
├── services/                  # jd_decomposition, retrieval, feature, trust, ranking, scenario, explainability
├── pipelines/                 # live_query_pipeline, batch_scoring_pipeline
├── data_access/                # candidate_repository, feature_store_repository
├── cache/jd_decompositions/    # SHA-256-keyed JD decomposition cache
├── reports/backend_architecture_spec.md
└── tests/

frontend/
├── src/
│   ├── lib/SearchStateProvider.tsx   # shared client-side state: search, shortlist, latency, scenario interaction
│   ├── components/                   # CandidateCard, FeatureBar, TrustBadge, ScenarioExplorer, Navbar
│   └── app/
│       ├── page.tsx                  # Home
│       ├── discovery/page.tsx
│       ├── candidate/[id]/page.tsx
│       ├── compare/page.tsx
│       ├── shortlist/page.tsx
│       └── demo/page.tsx
└── tokens.css
```

## Screens

| Screen | Purpose |
|---|---|
| **Home** | Browser-new-tab-style entry point: an omnibox plus six suggestion tiles (role-based and natural-language). Tapping a tile runs a real search and previews the top 3 results inline — a judge never has to type anything to see a ranked result. |
| **Discovery** | Funnel stats, JD decomposition chips, the full ranked candidate list, and the entry point into the scenario explorer. |
| **Candidate detail** | The opened profile: SHAP-based "why this rank," skill gap with counterfactual, full five-check trust breakdown with the standing caveat, fingerprint badge (with caveat) if applicable. |
| **Scenario explorer** | Default vs. Custom weight mode. Default shows the primary GBM ranking with zero extra API calls; Custom unlocks six sliders and calls the linear-baseline rerank endpoint live. |
| **Compare** | 2–4 candidates side by side against the model's own features, with neutral-colored deltas against the top-ranked candidate. |
| **Shortlist** | A recruiter's saved subset, at their original (non-sequential) ranks. |
| **Demo mode** | A single highlight screen for presenting: live latency vs. the 300ms/800ms targets, the funnel, the current #1 candidate's top reason, and the most recent real scenario-explorer interaction. |

## API reference

| Endpoint | Purpose |
|---|---|
| `POST /search` | Run a JD against the candidate pool, return a ranked, explained shortlist. |
| `GET /candidate/{id}` | Full profile, trust breakdown, and SHAP attribution for one candidate. |
| `POST /compare` | 2–4 candidates side by side with a feature comparison matrix. |
| `POST /scenarios/rerank` | Recruiter-weighted linear re-rank within an existing search session. |
| `GET /health` | Model version, feature store size, cache stats, degraded-state flag. |

Full request/response shapes are in `backend/reports/backend_architecture_spec.md`
and mirrored in the frontend's `types.ts` — the two are kept in sync by hand;
there's no codegen step.

## Trust & transparency principles

These aren't backend implementation details — they're product decisions that
show up directly in the UI, and they're worth a judge's attention as much as
any feature:

1. Every trust score ships with what it can and can't detect, every time
   it's shown.
2. Every ranking is explained with the same SHAP values the model actually
   used, not a post-hoc summary.
3. A statistically suspicious signal (the fingerprint pattern) is disclosed
   and deliberately capped rather than quietly leveraged.
4. Near-identical scores are labeled as ties, not presented as a meaningful
   ordering.
5. Counterfactual skill-gap estimates are explicitly flagged as
   approximations, never as model predictions.

## Trust & Fairness Evidence

The system's trust scoring has been rigorously audited for both detection capability and potential bias. For full details, see the [trust_score_audit_report.md](file:///c:/projects/Veritalent/trust_lab/reports/trust_score_audit_report.md).
For a comprehensive breakdown of ranking quality and ablation studies, refer to our [EVALUATION_SUMMARY.md](file:///c:/projects/Veritalent/EVALUATION_SUMMARY.md).

**Detection Performance:**
- **Honeypot detection rate:** 70/80 known honeypots were caught.
- **False positives:** 9 candidates flagged (score >= 0.4). Characterization: These aren't malicious bots, but legitimate "over-claimers" who lack identity verification or scored poorly on assessments despite claiming advanced proficiency.
- **Explicit limitation statement:** "The current trust score catches sloppy/inconsistent fraud; it **does NOT reliably catch sophisticated, internally-consistent fraud**."

**Bias Analysis:**
There is no statistically significant correlation between the trust flags and nonlinear career indicators.
- **Correlation with Career Gaps:** 0.000
- **Correlation with Industry Switches:** -0.007
- **Correlation with Technical Degree:** -0.005

## Setup

### Backend

```bash
cd backend
pip install fastapi uvicorn lightgbm shap cachetools sentence-transformers
```

Set one of these (optional — the system runs fully offline without it, via
the keyword-extraction fallback for JD decomposition):

```bash
export CEREBRAS_API_KEY=...   # or
export GROQ_API_KEY=...
```

```bash
uvicorn backend.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
```

Point the frontend at your backend (defaults to `http://localhost:8000`):

```bash
# .env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

```bash
npm run dev
```

## Running tests

```bash
# Backend
pytest backend/tests/ -v

# Frontend
npm run lint
npm run build
```

`npm run build` and `npm run lint` are expected to exit clean with zero
errors and warnings. On the frontend, `test_fingerprint_badge_caveat_present.js`
asserts the fingerprint caveat renders verbatim whenever the badge does, and
is absent whenever it doesn't.
