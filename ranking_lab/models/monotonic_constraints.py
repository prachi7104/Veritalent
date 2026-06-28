# ranking_lab/models/monotonic_constraints.py

TRAINING_FEATURES = [
    "skill_depth",
    "skill_breadth",
    "skill_recency",
    "skill_mastery_triangulation",
    "jd_skill_score",
    "tenure_stability",
    "activity_quality_composite",
    "trust_score",
    "logistics_fit_score",
    "product_vs_services",
    "implied_skill_score",
    "yoe_band_fit"
]

def get_monotonic_constraints() -> list[int]:
    return [
        1,   # skill_depth
        0,   # skill_breadth
        1,   # skill_recency
        0,   # skill_mastery_triangulation
        1,   # jd_skill_score
        0,   # tenure_stability
        0,   # activity_quality_composite
        -1,  # trust_score
        0,   # logistics_fit_score
        0,   # product_vs_services
        1,   # implied_skill_score
        1,   # yoe_band_fit (higher band fit → higher rank)
    ]

if __name__ == "__main__":
    print(f"Features: {TRAINING_FEATURES}")
    print(f"Constraints: {get_monotonic_constraints()}")
