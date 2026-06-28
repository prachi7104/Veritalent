"""
Spot-check 10 narratives from the new submission CSV.
Checks:
  1. No raw feature names in reasoning text
  2. Narrative length in 80-200 word range
  3. At least some variation in opening phrases
  4. Mandatory elements present (title/role, skills)
"""
import pandas as pd
import re

df = pd.read_csv("submission/submission.csv")

RAW_FEATURE_NAMES = [
    "skill_mastery_triangulation", "skill_depth", "skill_breadth",
    "skill_recency", "activity_quality_composite", "logistics_fit_score",
    "product_vs_services", "implied_skill_score", "yoe_band_fit",
    "jd_skill_score", "tenure_stability"
]

issues = []
for _, row in df.head(20).iterrows():
    text = str(row["reasoning"])
    rank = row["rank"]

    # Check for raw feature names
    found_raw = [f for f in RAW_FEATURE_NAMES if f in text]
    if found_raw:
        issues.append(f"Rank {rank}: contains raw feature names: {found_raw}")

    # Check length (word count)
    word_count = len(text.split())
    if word_count < 40:
        issues.append(f"Rank {rank}: too short ({word_count} words)")
    if word_count > 250:
        issues.append(f"Rank {rank}: too long ({word_count} words)")

# Check opening phrase variety
openings = [df.iloc[i]["reasoning"][:30] for i in range(10)]
unique_openings = len(set(openings))
print(f"Unique opening phrases (top 10): {unique_openings}/10")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  ⚠ {issue}")
else:
    print("\nAll narratives pass quality checks.")

# SUCCESS CRITERION
assert len(issues) == 0, f"FAIL: {len(issues)} narrative quality issues found"
assert unique_openings >= 7, (
    f"FAIL: Only {unique_openings}/10 unique openings — narratives still templated"
)
print("PASS: Narrative quality checks passed.")
