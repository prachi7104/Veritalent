# Fingerprint Validation Lab

Adversarial stress-test of the Tier-5 Fingerprint Radar: 13 ultra-rare skill strings,
8 candidates, all Senior/Staff/Lead AI/ML titled. The job is to try to **disprove** its usefulness.

---

## Directory Structure

```
fingerprint_lab/
├── analysis/
│   ├── frequency_audit.py          # Independent re-derivation from scratch
│   ├── redundancy_check.py         # Signal vs. title+deep-IR only
│   ├── boundary_sensitivity.py     # Cutoff stability test
│   ├── leakage_hypothesis_test.py  # Planted artifact investigation
│   └── marginal_value_test.py      # NDCG delta measurement
├── reports/
│   ├── frequency_audit_results.json
│   └── fingerprint_validation_report.md  ← Final verdict here
└── tests/
    └── test_fingerprint.py         # 15 tests, all passing
```

---

## Quick Start

### Prerequisites
Feature store must be generated first:
```bash
python -m feature_lab.store.feature_store \
    --input  "dataset/.../candidates.jsonl" \
    --output "feature_lab/store/feature_store.jsonl"
```

### Run all analyses in order

```bash
# 1. Independent frequency audit (generates reports/frequency_audit_results.json)
python -m fingerprint_lab.analysis.frequency_audit \
    --input  "dataset/.../candidates.jsonl" \
    --output "fingerprint_lab/reports/frequency_audit_results.json"

# 2. Boundary sensitivity (uses candidates.jsonl directly)
python -m fingerprint_lab.analysis.boundary_sensitivity \
    --input "dataset/.../candidates.jsonl"

# 3. Redundancy check (requires audit results)
python -m fingerprint_lab.analysis.redundancy_check \
    --input "dataset/.../candidates.jsonl" \
    --audit "fingerprint_lab/reports/frequency_audit_results.json"

# 4. Leakage hypothesis test (requires audit results, fast)
python -m fingerprint_lab.analysis.leakage_hypothesis_test \
    --audit "fingerprint_lab/reports/frequency_audit_results.json"

# 5. Marginal value test (requires feature store + audit)
python -m fingerprint_lab.analysis.marginal_value_test \
    --store "feature_lab/store/feature_store.jsonl" \
    --audit "fingerprint_lab/reports/frequency_audit_results.json"

# 6. Tests
pytest fingerprint_lab/tests/test_fingerprint.py -v
```

---

## Key Results (2026-06-23)

| Analysis | Finding |
|----------|---------|
| Frequency re-derivation | 13/13 skills confirmed, 8/8 holders, 100% Senior AI/ML alignment |
| Redundancy check | 6/8 already in top 50 without fingerprint flag |
| Boundary sensitivity | Candidate set perfectly stable from threshold 5–20 |
| Leakage hypothesis | **LIKELY PLANTED** — high confidence |
| Marginal NDCG | delta = −0.0091 (flat bonus hurts aggregate NDCG) |

**Final Decision: MODIFY** — tiebreaker-only, ≤2% max contribution, never a GBM feature, always disclosed with n=8 caveat.

See [fingerprint_validation_report.md](reports/fingerprint_validation_report.md) for full findings.
