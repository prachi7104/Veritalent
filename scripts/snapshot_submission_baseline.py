# scripts/snapshot_submission_baseline.py
import pandas as pd
import json
from pathlib import Path
import datetime

df = pd.read_csv("submission/submission_baseline.csv"
                 if Path("submission/submission_baseline.csv").exists()
                 else "submission/submission.csv")

feature_mentions = {}
features = [
    "skill_mastery_triangulation", "skill_depth", "logistics_fit_score",
    "activity_quality_composite", "skill_breadth", "implied_skill_score",
    "trust_score", "tenure_stability", "skill_recency", "product_vs_services"
]
for feat in features:
    feature_mentions[feat] = int(df["reasoning"].str.contains(feat).sum())

report = {
    "total_candidates": len(df),
    "score_mean": round(df["score"].mean(), 4),
    "score_std": round(df["score"].std(), 4),
    "score_min": round(df["score"].min(), 4),
    "score_max": round(df["score"].max(), 4),
    "n_tied_scores": int(len(df) - df["score"].round(4).nunique()),
    "reasoning_length_mean": round(df["reasoning"].str.len().mean(), 1),
    "reasoning_length_min": int(df["reasoning"].str.len().min()),
    "feature_mention_counts": feature_mentions,
    "dominant_feature": max(feature_mentions, key=feature_mentions.get),
    "dominant_feature_rate": round(
        feature_mentions[max(feature_mentions, key=feature_mentions.get)] / len(df), 3
    ),
}

out_path = Path(f"reports_archive/baseline_{datetime.date.today().strftime('%Y%m%d')}"
                "/submission_baseline_stats.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

for k, v in report.items():
    print(f"{k}: {v}")
