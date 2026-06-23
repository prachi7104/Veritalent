# Fingerprint Validation Report

**Lab:** `fingerprint_lab`  
**Date:** 2026-06-23  
**Purpose:** Adversarial stress-test of the Tier-5 Fingerprint Radar concept.  
**Approach:** Try to DISPROVE the signal's usefulness. All analyses run against the full 100,000-candidate pool — nothing is extrapolated from a sample.

---

## 1. Independent Frequency Re-Derivation

**Method:** Full independent re-count of every skill string across all 100,000 candidates from scratch. No cached values reused.

### Results

| Metric | Claimed | Independently Verified |
|--------|---------|----------------------|
| Total unique skill strings | 133 | **133** ✓ |
| Ultra-rare skills (1–7 occurrences) | 13 | **14** ⚠️ |
| Fingerprint skills found | 13 | **13/13** ✓ |
| Fingerprint holders | 8 | **8** ✓ |
| Senior AI/ML alignment | 100% | **8/8 = 100.0%** ✓ |

> [!WARNING]
> **Minor discrepancy flagged:** The ultra-rare band contains **14** skill strings, not 13 as implied in prior documentation. The 14th ultra-rare skill is not one of the 13 documented fingerprint strings — it is a different skill at ≤7 occurrences. This does not affect the fingerprint analysis, but it is reported explicitly as required.

**Per-fingerprint-skill occurrence counts (sorted ascending):**

| Skill | Unique Candidates |
|-------|:-----------------:|
| Search Backend | 1 |
| Content Matching | 3 |
| Text Encoders | 3 |
| Model Adaptation | 3 |
| Workflow Orchestration | 3 |
| Search Infrastructure | 4 |
| Search & Discovery | 4 |
| Indexing Algorithms | 4 |
| Vector Representations | 4 |
| Information Retrieval Systems | 4 |
| Open-source ML libraries | 5 |
| Ranking Systems | 5 |
| Document Processing | 7 |

**The 8 fingerprint holders and their titles:**

| Candidate ID | Title | Senior AI/ML? |
|--------------|-------|:-------------:|
| CAND_0080766 | Staff Machine Learning Engineer | ✓ |
| CAND_0068351 | Lead AI Engineer | ✓ |
| CAND_0030468 | Senior Applied Scientist | ✓ |
| CAND_0037980 | Senior Applied Scientist | ✓ |
| CAND_0005538 | Senior AI Engineer | ✓ |
| CAND_0093193 | Senior Machine Learning Engineer | ✓ |
| CAND_0006567 | Senior AI Engineer | ✓ |
| CAND_0061257 | Staff Machine Learning Engineer | ✓ |

**Conclusion:** The claimed finding is confirmed with one minor discrepancy (14 ultra-rare skills, not 13 in the wider pool). All 13 documented fingerprint skills are present. All 8 holders are Senior/Staff/Lead AI/ML titled.

---

## 2. Redundancy Check

**Method:** Score all 100,000 candidates using only `title_tier + deep-IR skill depth` (no fingerprint flag). Check where the 8 fingerprint holders land in that ranking.

| Candidate | Title | Rank (no FP) | Rank (with FP bonus) | Rank Change | Percentile |
|-----------|-------|:------------:|:--------------------:|:-----------:|:----------:|
| CAND_0080766 | Staff ML Engineer | **1** | 1 | 0 | 100.0% |
| CAND_0068351 | Lead AI Engineer | **3** | 2 | +1 | 100.0% |
| CAND_0030468 | Senior Applied Scientist | **6** | 3 | +3 | 100.0% |
| CAND_0037980 | Senior Applied Scientist | **17** | 4 | +13 | 100.0% |
| CAND_0005538 | Senior AI Engineer | **19** | 5 | +14 | 100.0% |
| CAND_0093193 | Senior ML Engineer | **26** | 6 | +20 | 100.0% |
| CAND_0006567 | Senior AI Engineer | **70** | 8 | +62 | 99.9% |
| CAND_0061257 | Staff ML Engineer | **116** | 25 | +91 | 99.9% |

**Summary:**
- **6 of 8** fingerprint holders are already in the top 50 without the fingerprint flag
- **8 of 8** are in the top 200 without the flag
- **8 of 8** are in the top 500 without the flag

**Conclusion:** The fingerprint signal is almost entirely redundant with signals the system already has (title seniority + deep-IR skill depth). Six of eight fingerprint holders reach the top 50 with no fingerprint bonus at all. The two "rescued" by the fingerprint flag (CAND_0006567 at rank 70→8, CAND_0061257 at rank 116→25) represent genuine lift — but both were already in the top 0.1% of the 100k pool without the flag. The marginal information content of the fingerprint, relative to what the system already knows, is low.

---

## 3. Boundary Sensitivity Test

**Method:** Redefine "ultra-rare" at thresholds ≤3, ≤5, ≤7, ≤10, ≤15, ≤20. Measure stability of the fingerprint candidate set.

| Cutoff | FP Skills Qualifying | Holders | Senior AI/ML | Alignment | Core Stability |
|--------|:--------------------:|:-------:|:------------:|:---------:|:--------------:|
| ≤3 | 5 | 7 | 7 | 100.0% | 7/8 retained |
| ≤5 | 12 | 8 | 8 | 100.0% | 8/8 retained |
| ≤7 | 13 | 8 | 8 | 100.0% | 8/8 retained *(baseline)* |
| ≤10 | 13 | 8 | 8 | 100.0% | 8/8 retained |
| ≤15 | 13 | 8 | 8 | 100.0% | 8/8 retained |
| ≤20 | 13 | 8 | 8 | 100.0% | 8/8 retained |

**Critical finding:** The candidate set is **perfectly frozen** from threshold ≤5 through ≤20. No new candidates enter at higher thresholds because there are simply no skill strings with 8–20 occurrences among the fingerprint skills — every skill is either ≤7 or >20. The holder set is also perfectly stable: 7 of 8 core candidates appear even at the tightest cutoff (≤3), and all 8 appear at ≤5 onwards.

**Interpretation:** This is anomalously stable. A natural power-law skill tail would produce gradual inclusion of new candidates and occasional title mismatches as thresholds widen. The complete stability across 5× range of thresholds (5 → 20), combined with zero contamination by non-Senior-AI/ML candidates at any threshold, is inconsistent with organic, noisy data. It is consistent with a deliberately curated set.

---

## 4. Leakage Hypothesis Test

**Method:** Three independent tests.

### Test 1 — JD Textual Proximity

Each fingerprint skill string was scored for how directly it paraphrases the target JD's language (score 0 = generic, 1.0 = directly echoes JD phrases).

| Skill | Suspicion Score | Key JD Phrase Matches |
|-------|:--------------:|----------------------|
| Search Backend | **1.00** | "search", "backend" |
| Text Encoders | **1.00** | "text", "encoder" |
| Vector Representations | **1.00** | "vector", "representation" |
| Search & Discovery | **1.00** | "search", "discovery" |
| Search Infrastructure | **1.00** | "search", "infrastructure" |
| Indexing Algorithms | **1.00** | "indexing", "algorithm" |
| Workflow Orchestration | **1.00** | "workflow", "orchestration" |
| Information Retrieval Systems | **0.67** | "retrieval", "information retrieval" |
| Ranking Systems | **0.50** | "ranking" |
| Model Adaptation | **0.50** | "model" |
| Document Processing | **0.50** | "document" |
| Content Matching | **0.33** | "matching" |
| Open-source ML libraries | **0.33** | "open-source", "library" |

**Average suspicion score: 0.725.** 11 of 13 skills score ≥0.5. Seven skills score a perfect 1.0 — every word in the skill string directly maps to JD vocabulary. This is not what organic skill-listing behavior looks like. Organically accumulated skills tend to be tool names ("LangChain", "Elasticsearch"), framework names, or job-function labels — not paraphrased requirements documents.

### Test 2 — Seniority Alignment Under Null Hypothesis

- **Alignment rate: 100.0% (8/8)**
- Pool base rate of Senior AI/ML candidates: ~0.13% (130 of 100,000)
- P(all 8 holders are Senior AI/ML by random chance) = (0.0013)^8 = **8.16 × 10⁻²⁴**

This probability is essentially zero. Under any reasonable organic sampling process, at least one or two ultra-rare skill holders would be non-Senior-AI/ML candidates. The perfect alignment is not statistically compatible with a randomly occurring pattern.

### Test 3 — Occurrence Count Distribution

Occurrence counts: [1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 7]  
Mean: 3.85, Std: 1.35, **CV (coefficient of variation): 0.351**

A natural power-law tail (Zipfian distribution) would show CV >> 1, with a few skills at 1–2 occurrences and rapid falloff. A CV of 0.351 indicates a **suspiciously clustered** distribution — the counts are tightly bunched in the 3–7 range, with no outliers. This is more consistent with a set that was constructed to be small (to create a radar signal) than with organic accumulation.

### Leakage Verdict

> [!CAUTION]
> **LIKELY PLANTED — HIGH CONFIDENCE**
>
> All three independent tests converge on the same conclusion:
>
> 1. **Textual evidence (strong):** 7 of 13 skill strings are verbatim paraphrases of JD requirement phrases. Average JD overlap score = 0.725. Organic skill vocabularies do not echo JD language this precisely.
>
> 2. **Statistical impossibility (decisive):** P(8/8 senior alignment by chance) = 8.16 × 10⁻²⁴. This rules out organic coincidence.
>
> 3. **Distribution anomaly (supporting):** Occurrence counts form a tight cluster (CV=0.35) inconsistent with natural power-law skill tails.
>
> **Assessment:** These skills were almost certainly placed in the dataset by the challenge authors as a deliberately constructed evaluation signal — either as a breadcrumb for sophisticated teams to find, or as a ranking quality benchmark. They are not organically accumulated skills reflecting real-world candidate behavior. This does not make them useless (they still correlate with the candidates the system should rank highly), but it fundamentally changes what the signal means: it is *target leakage in mild form*, not *causal evidence of candidate quality*.
>
> **Important nuance:** The 8 candidates themselves may be entirely legitimate high-quality candidates — the issue is that the skill strings are synthetic markers, not that the candidates are fabricated. Using the fingerprint to rank these candidates higher is ranking them correctly for the wrong reason.

---

## 5. Marginal Value Test

**Method:** Linear weighted scorer (same as feature_lab ablation) + fixed fingerprint bonus (0.1 normalized units) vs. same scorer without bonus. Measured against `gold_set_pooled.json`.

| Condition | NDCG@50 |
|-----------|:-------:|
| Without fingerprint bonus | **0.4102** |
| With fingerprint bonus | **0.4011** |
| **Delta** | **−0.0091** |

**The fingerprint bonus *hurts* NDCG by 0.0091.**

**Root cause:** 5 of 8 fingerprint holders are not in the gold set at all. The bonus promotes these 5 non-relevant candidates into the top positions, displacing candidates who do appear in the gold set. In the ranking at bonus-threshold, CAND_0068351 moves to rank 1, CAND_0080766 to rank 2, CAND_0030468 to rank 3 — none of them are in the gold set — which degrades NDCG@50 relative to the baseline.

**Gold set coverage of fingerprint holders:**

| Candidate | Gold Relevance | Rank (no bonus) | Rank (with bonus) |
|-----------|:--------------:|:---------------:|:-----------------:|
| CAND_0061257 | **3** (excellent) | 17 | 6 |
| CAND_0006567 | **3** (excellent) | 12 | 5 |
| CAND_0037980 | **2** (strong) | 6 | 4 |
| CAND_0068351 | NOT IN GOLD | 1 | 1 |
| CAND_0080766 | NOT IN GOLD | 3 | 2 |
| CAND_0030468 | NOT IN GOLD | 5 | 3 |
| CAND_0005538 | NOT IN GOLD | 25 | 8 |
| CAND_0093193 | NOT IN GOLD | 28 | 9 |

Note: the 3 gold-relevant fingerprint holders are already well-ranked without the bonus (ranks 6, 12, 17). The fingerprint bonus rescues them marginally in rank (6→4, 12→5, 17→6) but at the cost of inflating 5 non-relevant candidates above them.

**Honest interpretation:** The aggregate NDCG delta is −0.0091, which falls within the low-marginal-value range (|Δ| < 0.01) defined in the feature_lab ablation. The fingerprint as a *flat bonus to all 8 holders* is counterproductive. As a *conditional tiebreaker only among otherwise equally-ranked top candidates*, the harm would be smaller — but n=8 means this would almost never fire in practice.

---

## 6. Final Decision

> [!IMPORTANT]
> ## FINAL DECISION: MODIFY
>
> **Not REMOVE** — because the 3 fingerprint holders in the gold set are all high-relevance (scores 2 and 3), and the fingerprint correctly identifies them. The signal is directionally correct.
>
> **Not KEEP as-is** — because: (a) it is very likely a planted artifact, not an organic quality signal; (b) it adds negative NDCG when applied as a flat bonus; (c) it is redundant with existing signals (6/8 holders are already top-50 without it); (d) n=8 is too small for any reliable weight-learning.
>
> ### Exact permitted usage (bounded policy)
>
> The fingerprint flag may be used **only** as follows:
>
> 1. **Capped tiebreaker only:** Applied exclusively to break ties between candidates with near-identical scores from the primary feature set (skill, career, activity). Never as a standalone boosting signal.
>
> 2. **Maximum contribution:** The fingerprint flag must contribute at most **1–2% of the final composite score**. This must be enforced as a hard cap in code, not left to learned GBM weights.
>
> 3. **Not a GBM training feature:** Do not include `fingerprint_flag` as an input to the Lab 06 LightGBM model. With n=8 positive examples across 100k, it cannot learn a reliable weight — it will either be ignored or overfit. If included, its learned contribution must be manually verified to be near-zero.
>
> 4. **Presentation:** Always disclosed to judges and recruiters with explicit small-sample caveat. Never described as a "proven quality signal."
>
> 5. **Re-evaluate at Lab 07:** If real human-annotated labels become available, re-run the marginal value test. The current negative delta is driven by gold-set construction, which may not capture the full set of relevant fingerprint holders.

---

## 7. Recommended Demo / Documentation Language

### Acceptable wording (use verbatim or as close to this as possible):

> *"During dataset analysis, we discovered an ultra-rare vocabulary pattern: 13 skill strings appearing in 1–7 candidates each, all held exclusively by Senior/Staff/Lead AI/ML engineers — a pattern present in only 8 of 100,000 candidates. The alignment is statistically striking (p < 10⁻²³ under a null model), but we cannot rule out that these skill strings were deliberately placed in the dataset as an evaluation marker rather than reflecting organic candidate behavior.*
>
> *We treat this pattern as a capped secondary tiebreaker — it can nudge an otherwise equally-ranked candidate upward, but it is never a primary ranking driver and contributes at most 1–2% of any candidate's final score. We disclose this explicitly because using a potentially synthetic signal as a primary feature, without caveat, would violate the honesty principle guiding this system's design."*

### Unacceptable wording (do not use in any form):

- "We discovered a hidden expert fingerprint that uniquely identifies the top candidates"
- "The Tier-5 Fingerprint Radar is a novel signal we developed that identifies elite candidates"
- Any description that presents this as a causally-grounded discovery about candidate quality
- Any description that omits the small-sample-size caveat (n=8) or the planting hypothesis

---

## 8. Summary of Evidence

| Test | Finding | Implication |
|------|---------|-------------|
| Frequency re-derivation | 13/13 skills confirmed, 8/8 holders, 100% alignment | Claimed pattern is real; minor discrepancy (14 ultra-rare, not 13) noted |
| Redundancy check | 6/8 already top-50 without fingerprint | Low marginal information content vs. existing features |
| Boundary sensitivity | Perfectly stable from threshold 5–20, no contamination | Anomalously clean — inconsistent with organic noise |
| Leakage hypothesis | Avg suspicion 0.725, P(null) = 8e-24, CV = 0.35 | **Likely planted**; high confidence |
| Marginal value (NDCG) | Delta = −0.0091 | Flat bonus hurts aggregate NDCG; tiebreaker-only use warranted |
