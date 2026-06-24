import sys
import os
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

FEATURE_GROUPS = {
    "skill":     ["skill_depth", "skill_breadth", "skill_recency", "skill_mastery_triangulation"],
    "career":    ["tenure_stability", "promotion_velocity", "inflection_point_strength"], # Excluded career_velocity
    "trust":     ["trust_score"],
    "activity":  ["activity_quality_composite"],
    "industry":  [], # Excluded industry_relevance based on ablation
    "logistics": ["logistics_fit_score"],
    "company":   ["product_vs_services"],
}

GROUP_WEIGHTS = {
    "skill":     0.30,
    "career":    0.15,
    "activity":  0.20,
    "company":   0.10,
    "trust":     0.10, # trust score contributes via linear baseline as a fallback
    "logistics": 0.05,
    "industry":  0.00
}

def normalize_features(candidates: dict):
    min_vals = defaultdict(lambda: float("inf"))
    max_vals = defaultdict(lambda: float("-inf"))

    for feats in candidates.values():
        for group_feats in FEATURE_GROUPS.values():
            for feat in group_feats:
                val = float(feats.get(feat, 0.0) or 0.0)
                if val < min_vals[feat]:
                    min_vals[feat] = val
                if val > max_vals[feat]:
                    max_vals[feat] = val

    for feats in candidates.values():
        for group_feats in FEATURE_GROUPS.values():
            for feat in group_feats:
                val = float(feats.get(feat, 0.0) or 0.0)
                denom = max_vals[feat] - min_vals[feat]
                # Special handling for trust score in linear fallback:
                # In feature store, trust_score is higher for higher risk.
                # Linear baseline assumes higher is better, so we invert it.
                norm_val = (val - min_vals[feat]) / denom if denom > 0 else 0.0
                if feat == "trust_score":
                    norm_val = 1.0 - norm_val
                feats[f"{feat}_norm"] = norm_val

def generate_synthetic_labels(feature_store_path: str, output_path: str):
    """
    Computes 0-3 relevance labels based on the hand-tuned linear baseline using normalized feature values and the weights from Lab 03.
    """
    candidates = {}
    with open(feature_store_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
            
    normalize_features(candidates)
    
    scored = []
    for cid, feats in candidates.items():
        score = 0.0
        for group, group_feats in FEATURE_GROUPS.items():
            g_sum = sum(feats.get(f"{f}_norm", 0.0) for f in group_feats)
            score += g_sum * GROUP_WEIGHTS.get(group, 0.0)
        scored.append((cid, score))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Bin into 0-3 labels: top 1% = 3, next 5% = 2, next 20% = 1, rest = 0
    n = len(scored)
    labels = {}
    for i, (cid, score) in enumerate(scored):
        if i < n * 0.01:
            lbl = 3
        elif i < n * 0.06:
            lbl = 2
        elif i < n * 0.26:
            lbl = 1
        else:
            lbl = 0
        labels[cid] = lbl
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Synthetic labels generated and saved to {output_path}")
    return labels

if __name__ == "__main__":
    generate_synthetic_labels(
        r"c:\projects\Veritalent\feature_lab\store\feature_store.jsonl",
        r"c:\projects\Veritalent\ranking_lab\labels\synthetic_formula_labels.json"
    )
