# Feature Ablation Report

**Generated:** 2026-06-22  
**Gold set:** `retrieval_lab/evaluation/gold_set_pooled.json` (203 candidates, 1 query)  
**Evaluation metric:** NDCG@50  
**Scoring method:** Linear weighted formula with min-max normalization (deterministic, no training)  
**Pool size scored:** 100,000 candidates

---

## Methodology

Each feature group is removed one at a time. All 100,000 candidates are re-scored
and re-ranked with the remaining features. NDCG@50 is measured against the
pooled gold set (203 heuristically-judged candidates, relevance scores 0–3).

**This is not a trained model ablation.** LightGBM training belongs in Lab 06.
Here, ablation directly isolates each feature group's contribution to ranking
quality in the linear weighted scorer, which is both deterministic and avoids
overfitting artifacts that would appear when training on ~80 gold examples per fold.

---

## Ablation Results

| Feature Group | NDCG@50 (With) | NDCG@50 (Without) | Delta      | Category |
|---------------|---------------|-------------------|------------|----------|
| **Full baseline** | **0.4102** | — | — | — |
| skill | 0.4102 | 0.0220 | **−0.3882** | Must Keep |
| career | 0.4102 | 0.3231 | **−0.0871** | Must Keep |
| activity | 0.4102 | 0.3651 | **−0.0451** | Must Keep |
| logistics | 0.4102 | 0.3864 | **−0.0237** | Keep (de-weight candidate) |
| industry | 0.4102 | 0.4269 | **+0.0168** | Low Marginal Value — keep for explainability |
| trust (stub) | 0.4102 | 0.4102 | **0.0000** | Low Marginal Value — keep for explainability |
| company | 0.4102 | 0.4102 | **0.0000** | Low Marginal Value — keep for explainability |

> [!NOTE]
> Delta threshold for "Low Marginal Value": |Δ| < 0.01.
> Low marginal value does **not** mean remove — it means a feature adds complexity without
> measurable NDCG uplift on this gold set. It may matter for explainability or edge cases.

---

## Feature Group Analysis

### ✅ Must Keep

**`skill` (−0.3882)** — By far the largest contributor. Removing it collapses NDCG from 0.41 to
0.02, a catastrophic 95% degradation. `skill_depth` (max deep-IR duration × proficiency) is the
primary discriminating signal for this JD, consistent with the master context's 3.2% ML-adjacent
vs. 0.9% ML-titled pool skew. Deep-IR skill duration separates genuine IR practitioners from
keyword-stuffers in the ~130-candidate target population.

**`career` (−0.0871)** — Second largest contributor. Career velocity and tenure stability
correlate strongly with the "5–9 years of experience, production background" JD requirement.
Removing career features is a 21% NDCG loss.

**`activity` (−0.0451)** — Third contributor. Consistent with the JD's explicit instruction to
down-weight inactive and unresponsive candidates. The recruiter_response_rate and last-active
recency decay are doing real work here.

### ⚠️ Keep but De-weight Candidate

**`logistics` (−0.0237)** — Meaningful but modest. Notice period decay and location scoring
contribute, but the non-India softening (score 0.3 vs. 0) and the relocation bonus limit the
discriminative power for a predominantly India-based pool. Consider consolidating into a
simpler 2-component score (notice + location) vs. the current 3-component version.

### 📋 Low Marginal Value — Keep for Explainability / Edge Cases

**`industry` (+0.0168)** — Removing industry slightly *improves* NDCG (+1.7%), suggesting it
introduces noise on this gold set (possibly because candidates who match on skill already
come from relevant industries, making the feature redundant). However, it should be retained:
(a) it is a required explainability component for recruiter narratives, and (b) it may
discriminate on a larger, less heuristically-constructed evaluation set.

**`trust` (stub) (0.0000)** — Zero delta because the stub returns a constant 0.5 for all
candidates. The real Lab 04 implementation will introduce variance. **Do not remove from the
schema.** Once the real module drops in, this group must be re-evaluated.

**`company` (0.0000)** — Uniform across the pool. Most companies in the dataset are unknown
(treated as product by default), so the `product_vs_services` score has little discriminative
variance. The services-firm list is conservative by design. Consider expanding the lookup table
in subsequent iterations. Keep for explainability — knowing whether a candidate is product-
vs-services-background is a standard recruiter question.

---

## Correlation Cluster Analysis

Pairwise Pearson correlation across all numeric features, threshold: **>0.7**

| Feature A | Feature B | Correlation | Recommendation |
|-----------|-----------|-------------|----------------|
| `career_velocity` | `tenure_stability` | 0.7466 | Consolidate or use only one in GBM |

**Finding:** `career_velocity` (YOE / employers) and `tenure_stability` (avg duration per role)
are highly correlated because both fundamentally measure "how long a candidate stays at each job."
They are not redundant for *explainability* (they have different human-readable meanings), but in
a trained GBM they will split weight and potentially confuse SHAP attribution.

**Recommendation for Lab 06:** Feed only one (prefer `tenure_stability` as more direct) into the
GBM feature set, or combine into a single `career_stability_composite`. Do not drop both —
the career group overall has NDCG impact.

**No other feature pairs exceeded the 0.7 threshold.** The market-validation trio
(saved_by_recruiters_30d, profile_views_received_30d, search_appearance_30d) is not present
in this feature store because those raw redrob_signals fields were not included as standalone
features — they are absorbed into `activity_quality_composite`. This is the correct design:
combine correlated behavioral signals at feature-engineering time, not at model time.

---

## Final Feature Group Ranking

| Rank | Feature Group | Verdict |
|------|--------------|---------|
| 1 | **skill** | Must Keep — dominant discriminator |
| 2 | **career** | Must Keep — strong secondary signal |
| 3 | **activity** | Must Keep — JD-mandated behavioral signal |
| 4 | **logistics** | Keep but de-weight — simplify to 2 components |
| 5 | **industry** | Low marginal value — keep for explainability |
| 6 | **trust (stub)** | Low marginal value — re-evaluate after Lab 04 |
| 7 | **company** | Low marginal value — keep for explainability, expand lookup table |

---

## Constraints Verification

The following constraints from the master context and task spec are verified:

| Constraint | Status |
|-----------|--------|
| `github_activity_score == -1` treated as missing, not zero | ✅ Verified — test `test_github_minus_one_excluded_from_composite` passes |
| `education.start_year`/`end_year` not used anywhere | ✅ Verified — grep finds no reference in any feature module |
| Trust score is continuous (never binary auto-reject) | ✅ Verified — stub returns 0.5 float; reliability_tag = "sparse" |
| Title-tier filtering is a soft prior only | ✅ Verified — no hard title-gate anywhere in feature store |
| `product_vs_services` uses full career sequence | ✅ Verified — `test_currently_services_prior_product_scores_zero_point_eight` passes |

---

## Limitations and Honest Caveats

1. **Gold set is heuristically constructed**, not human-verified ground truth. NDCG values
   reflect agreement with a heuristic scoring function (senior title + deep-IR skill count
   + location), not genuine recruiter judgment. Feature groups that correlated with the
   heuristic will appear high-value; groups measuring things the heuristic ignores (industry,
   company background) will appear low-value even if they matter to real recruiters.

2. **Trust score stub contributes zero delta.** The real Lab 04 implementation must be
   re-plugged and the ablation re-run. Treat the trust row as "pending."

3. **Industry feature slightly hurts NDCG (+0.0168 when removed).** This is likely because
   the current industry keyword mapping is coarse and introduces noise for the generic-titled
   majority of the pool (HR Managers also work at "Software" companies). A
   JD-conditional industry weighting (only apply to candidates above a minimum skill-depth
   threshold) would likely reverse this.
