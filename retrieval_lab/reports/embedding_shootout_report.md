# Embedding Shootout Report

## Results Table
| Model | Subsample NDCG |
| --- | --- |
| intfloat/e5-small-v2 | 0.5433 |
| BAAI/bge-small-en-v1.5 | 0.5147 |

---
## Production Validation Addendum

The shootout above was evaluated on a 3,255-candidate subsample.
Subsequent re-evaluation on the full 31K production corpus (the actual
deployment scenario) showed BAAI/bge-small-en-v1.5 outperforming
intfloat/e5-small-v2.

**Authoritative recommendation: BAAI/bge-small-en-v1.5**

Reason: full-corpus results override subsampled benchmark results when
they conflict. The e5-small prefix convention (query:/passage:) likely
produced a subsampled advantage that did not transfer to the full
corpus distribution.

Lesson: embedding model rankings from subsampled benchmarks should
always be validated against the full production corpus before the
default is changed in production code.
---
