# ranking_lab/models/monotonic_constraints.py

TRAINING_FEATURES = [
    "skill_depth",
    "skill_breadth",
    "skill_recency",
    "skill_mastery_triangulation",
    "tenure_stability",
    "activity_quality_composite",
    "trust_score",
    "logistics_fit_score",
    "product_vs_services",
    "implied_skill_score"
]

def get_monotonic_constraints() -> list[int]:
    constraints = []
    for feature in TRAINING_FEATURES:
        if feature == "trust_score":
            constraints.append(-1) # 1.0 = high concern risk flag, so higher score -> lower rank
        elif feature == "skill_depth":
            constraints.append(1)  # Higher depth -> higher rank
        elif feature == "implied_skill_score":
            constraints.append(1)  # Higher implied skill -> higher rank
        else:
            constraints.append(0)  # Let the model figure out the rest
    return constraints

if __name__ == "__main__":
    print(f"Features: {TRAINING_FEATURES}")
    print(f"Constraints: {get_monotonic_constraints()}")
