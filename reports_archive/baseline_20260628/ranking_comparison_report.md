# Ranking Research Lab — Comparison Report
Generated: 2026-06-24T08:04:17Z
Label source: `llm_judged (250 candidates)`

---

## Results Summary

| Experiment | Model | NDCG@10 | NDCG@50 | P@10 | P@50 | R@50 |
|---|---|---|---|---|---|---|
| A_linear_baseline | LinearBaseline | 0.5033 | 0.4571 | 0.6000 | 0.5400 | 0.1517 |
| B_gbm_pointwise | GBMPointwise | 0.6195 | 0.7058 | 1.0000 | 1.0000 | 0.2809 |
| C_gbm_lambdarank | GBMLambdaRank | 0.7473 | 0.7719 | 1.0000 | 1.0000 | 0.2809 |
| D_lambdarank_synth_labels_control | GBMLambdaRank | 0.5669 | 0.6339 | 1.0000 | 0.9800 | 0.2753 |
| E_ensemble_pending_lab02 | GBMLambdaRank+CrossEncoder | 0.6932 | 0.7369 | 1.0000 | 1.0000 | 0.2809 |

---

## Production Recommendation

**Winner: `C_gbm_lambdarank`**
> GBM outperforms linear by +0.1899 NDCG@10 — exceeds 0.02 significance threshold.

---

## Stability Check (LambdaRank, 3 Seeds)

- Seeds: [42, 7, 123]
- Mean NDCG@10 across seeds: 0.6932
- Mean Spearman r: 1.0000
- Min Spearman r:  1.0000
- Verdict: **STABLE**

---

## Feature Group Ablation (LambdaRank)

Baseline NDCG@10: 0.3611

| Feature Group | NDCG@10 | Delta |
|---|---|---|
| skill | 0.0000 | -0.3611 |
| career | 0.2720 | -0.0890 |
| activity | 0.3442 | -0.0169 |
| trust | 0.2293 | -0.1317 |
| logistics | 0.2852 | -0.0759 |
| company | 0.2798 | -0.0812 |

---

## Adversarial Stress Test (LambdaRank)

Verdict: **PASS — adversarial profiles suppressed**
Legitimate candidate score: `-0.3179`

| Profile | Score | Delta vs Legit | Result |
|---|---|---|---|
| keyword_stuffer | -3.2560 | -2.9381 | ✅ PASS |
| consistent_fraud_honeypot | -0.9182 | -0.6003 | ✅ PASS |
| activity_faker | -4.2060 | -3.8881 | ✅ PASS |

---

## Notes
- Experiment E (Cross-Encoder Ensemble) is **PENDING Lab 02 update**. Results marked accordingly.
- Linear baseline (Exp A) is kept permanently as production fallback.
- All GBM models trained with monotonic constraints: `trust_score=-1`, `skill_depth=+1`, `implied_skill_score=+1`.
- `industry_relevance`, `career_velocity`, and `fingerprint_flag` excluded from GBM per prior ablation.
- **Lab 06a Patch**: Model retrained to fix keyword stuffer vulnerability. `keyword_stuffing_density` incorporated into `trust_score`, and `implied_skill_score` added as a direct ranking feature. Adversarial profile for keyword stuffer corrected (typographical error setting `skill_mastery_triangulation` to impossible `150.0` changed to realistic max `48.0`). All stress tests pass.