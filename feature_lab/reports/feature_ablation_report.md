# Feature Ablation Report

**Generated:** 2026-06-23 (re-run; initial run 2026-06-22)
**Gold set:** `retrieval_lab/evaluation/gold_set_pooled.json` — **confirmed**
**Evaluation metric:** NDCG@50
**Scoring method:** Linear weighted formula with min-max normalization (deterministic, no training)
**Pool size scored:** 100,000 candidates
**Gold set size:** 203 candidates, relevance 0–3

---

## Gold Set Provenance — Explicitly Stated

> [!IMPORTANT]
> The ablation was run against **`gold_set_pooled.json`**, not `gold_set.json`.
> This is confirmed by the ablation runner's path-selection logic
> (`GOLD_SET_POOLED_PATH` takes priority; falls back to `gold_set.json` only if pooled is absent).
>
> **Implication for interpretation:** `gold_set_pooled.json` is itself a pooled
> heuristic set — built from title + deep-IR skill counts + location signals.
> Skill features are therefore *structurally aligned* with the heuristic and will
> show inflated NDCG deltas. This is noted under Limitations (see below).
> Despite this, the numbers are the best available without human-annotated labels.
> They should be treated as directional signals, not ground truth, in Lab 06.

---

## Methodology

Each feature group is removed one at a time. All 100,000 candidates are re-scored
and re-ranked with the remaining features. NDCG@50 is measured against the
pooled gold set.

**This is not a trained model ablation.** LightGBM training belongs in Lab 06.
Here, ablation directly isolates each feature group's contribution to the linear
weighted scorer, which is deterministic and avoids overfitting artifacts.

---

## Ablation Results

| Feature Group | NDCG@50 (With) | NDCG@50 (Without) | Delta      | Category |
|---------------|---------------|-------------------|------------|----------|
| **Full baseline** | **0.4102** | — | — | — |
| skill | 0.4102 | 0.0220 | **−0.3882** | Must Keep |
| career | 0.4102 | 0.3231 | **−0.0871** | Must Keep |
| activity | 0.4102 | 0.3651 | **−0.0451** | Must Keep |
| logistics | 0.4102 | 0.3864 | **−0.0237** | Keep (de-weight candidate) |
| industry | 0.4102 | 0.4269 | **+0.0168** | Low Marginal Value — keep for explainability only |
| trust (stub) | 0.4102 | 0.4102 | **0.0000** | Low Marginal Value — re-evaluate after Lab 04 |
| company | 0.4102 | 0.4102 | **0.0000** | Low Marginal Value — keep for explainability only |

> [!NOTE]
> Delta threshold for "Low Marginal Value": |delta| < 0.01.

---

## Feature Group Analysis

### Must Keep

**`skill` (−0.3882)** — By far the largest contributor. Removing it collapses NDCG from 0.41 to
0.02. `skill_depth` (deep-IR duration × proficiency) is the primary discriminating signal.
**Caveat:** This delta is partially inflated because the gold set itself uses deep-IR skill count
as a scoring heuristic, creating circular alignment with the skill features. The true signal
magnitude is real but the exact number should not be taken as absolute ground truth.

**`career` (−0.0871)** — Strong secondary contributor. Career velocity and tenure stability
correlate with "5–9 years production background." Removing career features is a 21% NDCG loss.

**`activity` (−0.0451)** — Third contributor. Consistent with the JD's explicit instruction to
down-weight inactive and unresponsive candidates.

### Keep but De-weight Candidate

**`logistics` (−0.0237)** — Meaningful but modest. Consider simplifying to 2 components
(notice period + location) vs. the current 3-component version.

---

## Industry Feature Finding — Lab 06 Flag

> [!IMPORTANT]
> **Removing industry features *improves* NDCG by +0.0168.**
>
> This is a clear empirical result: the hand-tuned industry relevance weights from the
> master context's architecture plan (which assumed AI/ML > Fintech > IT Services) are
> **adding noise, not signal**, at least against this gold set.
>
> **Root cause hypothesis:** Candidates who match on skill depth (the dominant signal)
> already tend to come from relevant industries — the industry feature is therefore
> redundant for the strong positives. For the large irrelevant majority, the coarse
> keyword matching adds false positives (e.g., HR Managers at "Software" companies
> scoring high on industry_relevance despite being irrelevant).
>
> **Lab 06 directive — carry this forward explicitly:**
> - Do **NOT** pre-assign a high weight to industry features in the GBM feature set.
> - Let the model learn (or not learn) the industry signal. The ablation proves the
>   prior assumption was wrong.
> - If included at all, use `industry_relevance` only as a conditional modifier
>   (e.g., apply only to candidates above a minimum `skill_depth` threshold) rather
>   than as a standalone feature. A JD-conditional gate prevents it from rewarding
>   irrelevant candidates who happen to work at tech companies.
> - Report this finding in the Lab 06 feature importance analysis and note that any
>   non-zero GBM weight learned for industry is informative precisely because it was
>   not hand-tuned.

---

## Correlation Cluster Analysis

Pairwise Pearson correlation across all numeric features, threshold: **>0.7**

| Feature A | Feature B | Correlation | Recommendation |
|-----------|-----------|-------------|----------------|
| `career_velocity` | `tenure_stability` | 0.7466 | See action below |

**Finding:** `career_velocity` (YOE / employers) and `tenure_stability` (avg duration per role)
both measure "how long a candidate stays at each job." They are correlated because they share
the same underlying construct.

> [!IMPORTANT]
> **Resolved action for Lab 06 (explicit, carry forward):**
>
> - Feed **`tenure_stability` only** into the Lab 06 GBM as the ranking feature.
>   It is the more direct measure (duration per role, in months) and less sensitive
>   to the YOE self-report field (which has a 7.4% inconsistency rate).
> - Retain **`career_velocity` as an explainability/display feature only** — exposed
>   in recruiter-facing narratives ("has averaged X years per employer") but not as
>   a GBM input. It reads naturally in human explanations even if it carries redundant
>   information for the model.
> - Do **not** drop `career_velocity` from the feature store — it belongs in SHAP
>   narratives in Lab 07.

No other feature pairs exceeded 0.7. The market-validation trio (saved_by_recruiters_30d,
profile_views_received_30d, search_appearance_30d) is absorbed into `activity_quality_composite`
at feature-engineering time — correct design, avoids correlated raw signals in the model.

---

## Final Feature Group Ranking

| Rank | Feature Group | GBM Input? | Verdict |
|------|--------------|-----------|---------|
| 1 | **skill** | Yes | Must Keep — dominant discriminator |
| 2 | **career** | Yes (tenure_stability only; career_velocity explainability-only) | Must Keep |
| 3 | **activity** | Yes | Must Keep — JD-mandated behavioral signal |
| 4 | **logistics** | Yes, lower weight | Keep but de-weight |
| 5 | **industry** | No pre-assigned weight — let GBM decide | Low marginal value; ablation proved prior was wrong |
| 6 | **trust (stub)** | Re-evaluate after Lab 04 | Low marginal value — re-evaluate |
| 7 | **company** | Yes (explainability-critical) | Low marginal value — keep, expand lookup table |

---

## Constraints Verification

| Constraint | Status |
|-----------|--------|
| `github_activity_score == -1` treated as missing, not zero | Verified — test `test_github_minus_one_excluded_from_composite` passes |
| `education.start_year`/`end_year` not used anywhere | Verified — no reference in any feature module |
| Trust score is continuous (never binary auto-reject) | Verified — stub returns 0.5 float |
| Title-tier filtering is a soft prior only | Verified — no hard title-gate in feature store |
| `product_vs_services` uses full career sequence | Verified — `test_currently_services_prior_product_scores_zero_point_eight` passes |
| Ablation run against `gold_set_pooled.json` | **Verified** — confirmed by re-run 2026-06-23 |

---

## Limitations and Honest Caveats

1. **Gold set is heuristically constructed**, not human-verified ground truth. NDCG values
   reflect agreement with a heuristic function (senior title + deep-IR skill count + location).
   Feature groups correlated with the heuristic will appear high-value; the skill group's
   −0.3882 delta is partially explained by structural alignment between skill_depth and the
   gold-set scoring formula. **These numbers are directional inputs to Lab 06, not finalized
   feature importance values.**

2. **Trust score stub contributes zero delta.** The real Lab 04 implementation must be
   re-plugged and the ablation re-run. Treat the trust row as "pending." Expect non-zero
   delta once variance is introduced.

3. **Industry feature hurts NDCG (+0.0168 when removed).** Confirmed finding. Do not
   pre-weight in Lab 06. See "Industry Feature Finding" section above for the full action.

4. **`career_velocity` ↔ `tenure_stability` correlation (0.7466).** Resolved: feed only
   `tenure_stability` into GBM; retain `career_velocity` for explainability. See correlation
   section above.
