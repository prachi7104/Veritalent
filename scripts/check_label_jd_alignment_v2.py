# scripts/check_label_jd_alignment_v2.py
import json
from scipy import stats

with open("ranking_lab/labels/llm_labels.json") as f:
    raw = json.load(f)
labels = {cid: int(v["label"] if isinstance(v, dict) else v) for cid, v in raw.items()}

store = {}
with open("feature_lab/store/feature_store.jsonl") as f:
    for line in f:
        row = json.loads(line)
        store[row["candidate_id"]] = row

paired = [(labels[cid], store[cid]) for cid in labels if cid in store]
lbl = [p[0] for p in paired]

print("=== Label vs Feature Spearman Correlations ===")
feats = ["jd_skill_score", "skill_mastery_triangulation", "yoe_band_fit",
         "activity_quality_composite", "skill_depth", "product_vs_services",
         "implied_skill_score"]
correlations = {}
for feat in feats:
    vals = [float(p[1].get(feat, 0) or 0) for p in paired]
    r, pval = stats.spearmanr(lbl, vals)
    correlations[feat] = r
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*" if pval < 0.05 else "")
    print(f"  {feat:<35} r={r:+.4f}  {sig}")

jd_r = correlations["jd_skill_score"]
mastery_r = correlations["skill_mastery_triangulation"]
print()
if mastery_r > jd_r:
    print(f"WARNING: CIRCULAR LABEL: mastery r={mastery_r:+.4f} > jd_skill r={jd_r:+.4f}")
    print("  Labels measured seniority, not JD fit. NDCG understates jd_skill value.")
    print("  Use Netflix/Aganitha spot-check as secondary evidence of correctness.")
else:
    print(f"OK: LABELS CAPTURED JD FIT: jd_skill r={jd_r:+.4f} > mastery r={mastery_r:+.4f}")
