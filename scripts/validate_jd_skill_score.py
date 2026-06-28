# scripts/validate_jd_skill_score.py
"""
Validate that jd_skill_score diverges meaningfully from skill_mastery_triangulation.
Run after rebuilding feature_store.
"""
import json
import numpy as np
from scipy import stats

skill_mastery = []
jd_skill = []
candidate_ids = []

with open("feature_lab/store/feature_store.jsonl") as f:
    for line in f:
        row = json.loads(line)
        sm = row.get("skill_mastery_triangulation", 0) or 0
        js = row.get("jd_skill_score", 0) or 0
        skill_mastery.append(float(sm))
        jd_skill.append(float(js))
        candidate_ids.append(row["candidate_id"])

sm_arr = np.array(skill_mastery)
js_arr = np.array(jd_skill)

corr, pval = stats.spearmanr(sm_arr, js_arr)
print(f"Spearman correlation (skill_mastery vs jd_skill_score): {corr:.4f}")
print(f"P-value: {pval:.2e}")

# Rank candidates by each metric
sm_rank = np.argsort(-sm_arr)
js_rank = np.argsort(-js_arr)

sm_top20 = set(sm_rank[:20])
js_top20 = set(js_rank[:20])
overlap = sm_top20 & js_top20
print(f"\nTop-20 overlap between metrics: {len(overlap)}/20")
print(f"Candidates in jd_skill top-20 but NOT in mastery top-20: "
      f"{len(js_top20 - sm_top20)}")

# SUCCESS CRITERIA
try:
    assert corr < 0.85, (
        f"FAIL: jd_skill_score too correlated with skill_mastery ({corr:.4f}). "
        "Not providing independent signal."
    )
    assert len(js_top20 - sm_top20) >= 3, (
        "FAIL: jd_skill_score top-20 is nearly identical to skill_mastery top-20. "
        "No meaningful reranking happening."
    )
    print("\nPASS: jd_skill_score provides meaningfully independent ranking signal.")
except AssertionError as e:
    print(str(e))

# Print who moved up
print("\nCandidates promoted by jd_skill_score (not in mastery top-20):")
for idx in (js_top20 - sm_top20):
    true_idx = list(js_rank[:20]).index(idx) + 1
    mastery_pos = list(sm_rank).index(idx) + 1
    print(f"  {candidate_ids[idx]}: jd_skill rank={true_idx}, mastery rank={mastery_pos}")
