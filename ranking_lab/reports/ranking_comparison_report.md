# Ranking Research Lab — Comparison Report
Generated: 2026-06-30T20:13:44Z
Label source: `llm_judged (250 candidates)`

---

## Results Summary

| Experiment | Model | NDCG@10 | NDCG@50 | P@10 | P@50 | R@50 |
|---|---|---|---|---|---|---|
| A_linear_baseline | LinearBaseline | 0.1458 | 0.3697 | 0.3000 | 0.5400 | 0.1517 |
| B_gbm_pointwise | GBMPointwise | 0.4918 | 0.6427 | 1.0000 | 1.0000 | 0.2809 |
| C_gbm_lambdarank | GBMLambdaRank | 0.6771 | 0.7354 | 1.0000 | 1.0000 | 0.2809 |
| D_lambdarank_synth_labels_control | GBMLambdaRank | 0.4621 | 0.6477 | 1.0000 | 1.0000 | 0.2809 |
| E_ensemble_pending_lab02 | GBMLambdaRank+CrossEncoder | 0.6771 | 0.7354 | 1.0000 | 1.0000 | 0.2809 |

---

## Production Recommendation

**Winner: `C_gbm_lambdarank`**
> GBM outperforms linear by +0.5313 NDCG@10 — exceeds 0.02 significance threshold.

---

## Stability Check (LambdaRank, 3 Seeds)

- Seeds: [42, 7, 123]
- Mean NDCG@10 across seeds: 0.6771
- Mean Spearman r: 1.0000
- Min Spearman r:  1.0000
- Verdict: **STABLE**

---

## Feature Group Ablation (LambdaRank)

Baseline NDCG@10: 0.1558

| Feature Group | NDCG@10 | Delta |
|---|---|---|
| skill | 0.2835 | +0.1277 |
| career | 0.1873 | +0.0315 |
| activity | 0.2562 | +0.1004 |
| trust | 0.1389 | -0.0169 |
| logistics | 0.1909 | +0.0351 |
| company | 0.1558 | +0.0000 |

---

## Adversarial Stress Test (LambdaRank)

Verdict: **PASS**
Legitimate candidate score: `-2.4034`

| Profile | Score | Delta vs Legit | Result |
|---|---|---|---|
| keyword_stuffer | -4.2925 | -1.8892 | ✅ PASS |
| consistent_fraud_honeypot | -3.7704 | -1.3670 | ✅ PASS |
| activity_faker | -4.4904 | -2.0871 | ✅ PASS |

---

## Notes
- Experiment E (Cross-Encoder Ensemble) is **PENDING Lab 02 update**. Results marked accordingly.
- Linear baseline (Exp A) is kept permanently as production fallback.
- All GBM models trained with monotonic constraints: `trust_score=-1`, `skill_depth=+1`.
- `industry_relevance`, `career_velocity`, and `fingerprint_flag` excluded from GBM per prior ablation.