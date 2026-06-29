"""
Extract feature importances from the trained GBM that forms 90% of the blend.
This is what actually drives rankings — cite these in the PPT.
"""
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

model = GBMLambdaRankModel()
model.load("ranking_lab/models/gbm_lambdarank.txt")
imps = model.model.feature_importance()

print("=== Old GBM Feature Importances (90% of blend) ===")
print(f"{'Feature':<35} {'Importance':>12} {'% of Total':>12}")
print("-" * 62)
old_features = [
    "skill_depth", "skill_breadth", "skill_recency", 
    "skill_mastery_triangulation", "tenure_stability", 
    "activity_quality_composite", "trust_score", 
    "logistics_fit_score", "product_vs_services", 
    "implied_skill_score"
]
total = sum(imps)
for feat, imp in sorted(zip(old_features, imps), key=lambda x: -x[1]):
    pct = imp / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {feat:<33} {imp:>12.4f} {pct:>11.1f}%  {bar}")

print(f"\nTotal: {total:.4f}")
print("\nThese are the features you cite as 'primary ranking signals' in the PPT.")
print("jd_skill_score and yoe_band_fit are additive adjustments, not in this GBM.")
