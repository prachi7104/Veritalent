# Retrieval Comparison Report

## Fix 3 Addendum — Pooled Gold Set
- **Pooling Bias Confirmed:** The original gold set was built using BM25-like keyword heuristics. Semantic models (Dense, RRF) found valid candidates that lacked exact keywords and were therefore never labeled in the original gold set. When evaluated against that incomplete set, semantic methods were penalized for finding new, valid candidates (pooling bias).
- **The Fix:** A TREC-style pooled gold set was constructed by pooling the top 50 candidates from all baseline models. Pooling added 114 new positive cases, 65 of which were found ONLY by semantic models and were completely invisible to BM25.
- **C Anomaly NOT Resolved (Genuine Finding):** Even with pooling bias fixed, C (Hybrid RRF) NDCG@50 (0.3875) remains below max(A, B) (Dense B = 0.4344). **Root Cause:** With only 8% list overlap, RRF acts as an interleaver. Because Dense (B) is now correctly evaluated as significantly higher quality than BM25 (A), interleaving them 1:1 simply dilutes Dense's high-quality candidates with BM25's lower-quality candidates. RRF is mathematically incapable of improving over the best base model when overlap is this low and base model quality is this asymmetric.

### Final Recommendations (Based on Pooled Gold Set)
**Live Path (≤ 800ms):** **B: Dense Only** — NDCG@50 = 0.4345, Latency = ~89ms. (Note: With the unbiased gold set, Dense correctly dominates BM25. While G and E score slightly higher and fit the budget, Dense provides 92% of the maximum performance for just 11% of the latency cost, making it the clear ROI winner for the live path).
**Offline/Batch Path:** **G: Skill Graph Recall** — NDCG@50 = 0.4697 (highest overall quality, outperforming the generic Cross Encoder).

### Embedding Shootout (02) Configuration Inheritance
**CRITICAL:** The embedding shootout must evaluate against `gold_set_pooled.json`, not `gold_set.json`. Using the original gold set would bias quality metrics toward whichever embedding model behaves most similarly to BM25.

## Methodology
- Evaluated against `31179` candidates (allowlist domain funnel filter).
- Metrics evaluated at `k=10` and `k=50`.
- Live-path latency budget: 800ms.

## Results (Against Pooled Gold Set)
| Experiment                |   Old NDCG@50 |   NDCG@50 |   Delta NDCG |   P@10 |   P@50 |     R@50 |   Latency (ms) |
|:--------------------------|--------------:|----------:|-------------:|-------:|-------:|---------:|---------------:|
| A: BM25 Only              |     0.0527495 |  0.356116 |     0.303366 |    0.6 |   0.16 | 0.103896 |       458.393  |
| B: Dense Only             |     0.0338196 |  0.434462 |     0.400643 |    0.5 |   0.5  | 0.324675 |        89.4365 |
| C: Hybrid RRF             |     0.029965  |  0.387506 |     0.357541 |    0.4 |   0.4  | 0.25974  |       477.432  |
| D: Hybrid + Cross Encoder |     0.103005  |  0.432796 |     0.329791 |    0.7 |   0.48 | 0.311688 |     16184.1    |
| E: Hybrid Learned Fusion  |     0.0564812 |  0.45406  |     0.397578 |    0.5 |   0.54 | 0.350649 |       596.516  |
| F: Multi-Vector RRF       |     0.0898649 |  0.415008 |     0.325143 |    0.4 |   0.38 | 0.246753 |       830.945  |
| G: Skill Graph Recall     |     0.0632587 |  0.46971  |     0.406452 |    0.5 |   0.58 | 0.376623 |       780.273  |

## Qualitative Failure Notes
- **A: BM25 Only**: BM25 misses semantic variations and implicit experience.
- **B: Dense Only**: Dense embeddings can miss exact keyword requirements like BM25.
- **C: Hybrid RRF**: Prior poor performance was a pooling bias illusion. With pooled evaluation, Hybrid RRF functions correctly. However, due to low list overlap (~8%), it acts more like pure interleaving rather than true signal reinforcement.
- **D: Hybrid + Cross Encoder**: High latency due to cross-encoder inference. Limited to top 200, so recall cannot exceed stage 1.
- **E: Hybrid Learned Fusion**: Learned fusion is sensitive to the scale of raw scores and might overfit the gold set.
- **F: Multi-Vector RRF**: Requires structured query parsing; performance degrades if query mapping to fields is poor.
- **G: Skill Graph Recall**: Skill adjacency can pull in false positives if the co-occurrence is coincidental.
