# Feature Engineering Lab — Lab 06 Handoff Notes

These notes carry forward explicit, evidence-based directives from the feature
ablation results. They must be read before designing the Lab 06 LightGBM ranker.
Each directive is marked with its source evidence.

---

## 1. Industry features: do NOT pre-assign weight in GBM

**Evidence:** Removing `industry_relevance` from the linear scorer *improves*
NDCG@50 by +0.0168 (ablation run 2026-06-23 against `gold_set_pooled.json`).

**What this means:**
The hand-tuned industry relevance weights from the master context architecture
plan (AI/ML > Fintech > IT Services > unrelated) were a reasonable prior, but the
ablation proved they add noise rather than signal for this JD and this candidate pool.

The most likely cause: candidates who score high on `skill_depth` (deep-IR skills)
already tend to come from relevant industries, so the `industry_relevance` feature
is redundant for true positives. For the large irrelevant majority, it introduces
false positives (HR Managers at "Software" companies, etc.).

**Lab 06 action:**
- Do NOT set a non-zero initial weight or a monotonic constraint for `industry_relevance`
  in the GBM configuration.
- Let the LightGBM lambdarank objective assign whatever weight it learns from the
  labeled data. A low or zero learned weight is the correct outcome. A high weight
  is a red flag and should be investigated.
- If you want to include it conditionally, consider a threshold gate:
  only apply `industry_relevance` to candidates above a minimum `skill_depth`
  percentile (e.g., top 20%). This prevents it from rewarding the irrelevant majority.
- Report `industry_relevance` SHAP values explicitly in Lab 07. If the GBM learns
  near-zero weight, say so and cite this ablation finding. Do not bury it.

---

## 2. career_velocity vs tenure_stability: resolved split

**Evidence:** Pairwise Pearson correlation = 0.7466, exceeding the 0.7 consolidation
threshold from `correlation_check.py`. Both features measure the same underlying
construct ("how long a candidate stays at each job").

**Lab 06 action:**
- Feed **`tenure_stability` only** into the GBM as an input feature.
  - It is the more direct measure (average role duration in months).
  - It avoids dependence on `years_of_experience` (self-reported, 7.4% inconsistency rate).
- Retain **`career_velocity`** in the feature store but mark it as
  **explainability-only / display feature**:
  - Include it in SHAP-based recruiter narratives in Lab 07.
  - Do NOT pass it as a column in the GBM training DataFrame.
  - It reads naturally to recruiters: "has averaged X years per employer."
- Do NOT drop `career_velocity` from the store or schema — it belongs in narratives.

---

## 3. Gold set provenance — ablation numbers are directional, not ground truth

**Context:** Both ablation runs (2026-06-22 and 2026-06-23) used `gold_set_pooled.json`.
This is confirmed by the ablation runner's path-selection logic. The pooled set has
203 candidates with relevance 0–3, constructed heuristically from senior title +
deep-IR skill count + location signals.

**Implication for Lab 06 feature weighting:**
- The `skill` group's −0.3882 delta is partially inflated because the gold set itself
  uses deep-IR skill count as its heuristic — circular alignment overstates the
  apparent importance.
- The actual skill importance is still high (this JD genuinely requires deep IR skills),
  but the exact −0.3882 number should not be used as a multiplier for GBM weight tuning.
- Use these ablation numbers for **ranking feature groups by importance** (skill > career >
  activity > logistics > industry/trust/company), not for setting precise weights.
- Precise weight learning is the GBM's job. Provide stratified, diverse labels in Lab 06
  and let the model find the weights.

---

## Summary Directive Table for Lab 06

| Feature | GBM Input | Initial/Monotonic Constraint | Notes |
|---------|-----------|------------------------------|-------|
| `skill_depth` | Yes | Monotonic increasing | Primary discriminator |
| `skill_breadth` | Yes | None | Anti-keyword-stuffer signal |
| `skill_recency` | Yes | None | Binary, low variance — verify utility |
| `skill_mastery_triangulation` | Yes | None | Only non-zero for ~35.9% of pool |
| `tenure_stability` | Yes | None | Use instead of career_velocity |
| `promotion_velocity` | Yes | None | — |
| `inflection_point_strength` | Yes (experimental) | None | Tag as experimental in SHAP |
| `career_velocity` | **No** | — | Explainability/display only |
| `trust_score` | Yes | Monotonic increasing | Stub until Lab 04 drops in |
| `activity_quality_composite` | Yes | Monotonic increasing | JD-mandated signal |
| `industry_relevance` | **No pre-weight** | **No monotonic constraint** | Let GBM learn or ignore |
| `logistics_fit_score` | Yes | None | Lower weight candidate |
| `product_vs_services` | Yes | None | Expand lookup table first |
