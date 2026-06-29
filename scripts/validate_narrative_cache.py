# scripts/validate_narrative_cache.py
"""
Quality gate validation script for submission/narratives_cache.json.
Checks:
  1. No raw feature names in reasoning text
  2. Narrative length in 40-250 word range (target 80-120)
  3. At least 7 unique openings in top 10
  4. Blend annotation propagates (references JD/IR domain in >=30% of narratives)
"""
import json
from pathlib import Path

CACHE_PATH = Path("submission/narratives_cache.json")

RAW_FEATURE_NAMES = [
    "skill_mastery_triangulation", "skill_depth", "skill_breadth",
    "skill_recency", "activity_quality_composite", "logistics_fit_score",
    "product_vs_services", "implied_skill_score", "yoe_band_fit",
    "jd_skill_score", "tenure_stability", "recruiter_response_rate"
]

def run_validation():
    print("=== Validating Narrative Cache ===")
    assert CACHE_PATH.exists(), f"FAIL: {CACHE_PATH} does not exist"

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    print(f"Total narratives in cache: {len(cache)}")
    assert len(cache) == 100, f"FAIL: expected 100 narratives, got {len(cache)}"

    issues = []
    texts = []
    
    for idx, (cid, entry) in enumerate(cache.items()):
        text = entry["narrative"]
        texts.append(text)
        rank = idx + 1

        # Check for raw feature names
        found_raw = [f for f in RAW_FEATURE_NAMES if f in text]
        if found_raw:
            issues.append(f"Rank {rank} ({cid}): contains raw feature names: {found_raw}")

        # Check length
        word_count = len(text.split())
        if word_count < 40:
            issues.append(f"Rank {rank} ({cid}): too short ({word_count} words)")
        if word_count > 250:
            issues.append(f"Rank {rank} ({cid}): too long ({word_count} words)")

    # Check opening phrase variety (first 30 chars of top 10)
    openings = [texts[i][:30] for i in range(10)]
    unique_openings = len(set(openings))
    print(f"Unique opening phrases (top 10): {unique_openings}/10")

    # Check JD/IR domain references
    jd_ref_count = sum(
        1 for text in texts
        if any(phrase in text.lower() for phrase in
               ["jd alignment", "jd-relevant", "search", "retrieval", "ranking",
                "information retrieval", "vector", "embedding"])
    )
    print(f"Narratives referencing JD/IR domain: {jd_ref_count}/100")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("\nAll narratives pass format & length quality checks.")

    # Assertions
    assert len(issues) == 0, f"FAIL: {len(issues)} narrative quality issues found"
    assert unique_openings >= 7, (
        f"FAIL: Only {unique_openings}/10 unique openings — narratives still templated"
    )
    assert jd_ref_count >= 30, (
        f"FAIL: Only {jd_ref_count}/100 narratives reference JD domain. "
        "Blend annotation not propagating into narratives."
    )
    print("PASS: Narrative cache quality checks passed successfully!")


if __name__ == "__main__":
    run_validation()
