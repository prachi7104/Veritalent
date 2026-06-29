import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ranking_lab.models.gbm_pointwise import GBMPointwiseModel
from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

def create_dummy_data(n_samples=100):
    np.random.seed(42)
    n_features = len(TRAINING_FEATURES)
    X = np.random.rand(n_samples, n_features)
    # Simple linear combination for target
    y_cont = X.sum(axis=1) + np.random.normal(0, 0.1, n_samples)
    # Bin into integers 0-3 for lambdarank
    y = np.digitize(y_cont, bins=np.percentile(y_cont, [25, 50, 75]))
    return X, y

def test_monotonic_directions():
    print("Testing Monotonic Constraints Directions...")
    
    X_train, y_train = create_dummy_data()
    
    # We will test both models
    models = {
        "Pointwise": GBMPointwiseModel(random_state=42),
        "LambdaRank": GBMLambdaRankModel(random_state=42)
    }
    
    trust_score_idx = TRAINING_FEATURES.index("trust_score")
    skill_depth_idx = TRAINING_FEATURES.index("skill_depth")
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_train, y_train)
        
        print(f"Testing constraints on {name}...")
        
        # Create a base instance
        base_x = X_train[0].copy()
        
        # 1. Test trust_score constraint (-1 direction: increasing risk should NOT increase score)
        # We increase trust_score from 0.0 to 1.0 in increments of 0.1
        trust_scores = []
        x_test_trust = []
        for v in np.linspace(0.0, 1.0, 11):
            x_var = base_x.copy()
            x_var[trust_score_idx] = v
            x_test_trust.append(x_var)
            
        preds_trust = model.predict(np.array(x_test_trust))
        
        # Check that pred[i] >= pred[i+1] (or close due to float precision)
        is_monotonic_trust = True
        for i in range(len(preds_trust) - 1):
            if preds_trust[i] < preds_trust[i+1] - 1e-6: # allow tiny float tolerance
                is_monotonic_trust = False
                print(f"  [!] trust_score constraint FAILED: val={i/10.0} pred={preds_trust[i]} < val={(i+1)/10.0} pred={preds_trust[i+1]}")
                
        if is_monotonic_trust:
            print(f"  [PASS] trust_score constraint (-1) holds: as risk increases, rank decreases/plateaus.")
            
        # 2. Test skill_depth constraint (+1 direction: increasing depth should NOT decrease score)
        # We increase skill_depth from 0.0 to 100.0 in increments of 10.0
        depth_scores = []
        x_test_depth = []
        for v in np.linspace(0.0, 100.0, 11):
            x_var = base_x.copy()
            x_var[skill_depth_idx] = v
            x_test_depth.append(x_var)
            
        preds_depth = model.predict(np.array(x_test_depth))
        
        is_monotonic_depth = True
        for i in range(len(preds_depth) - 1):
            if preds_depth[i] > preds_depth[i+1] + 1e-6:
                is_monotonic_depth = False
                print(f"  [!] skill_depth constraint FAILED: val={i*10.0} pred={preds_depth[i]} > val={(i+1)*10.0} pred={preds_depth[i+1]}")
                
        if is_monotonic_depth:
            print(f"  [PASS] skill_depth constraint (+1) holds: as depth increases, rank increases/plateaus.")
            
        assert is_monotonic_trust, f"Model {name} failed trust_score constraint"
        assert is_monotonic_depth, f"Model {name} failed skill_depth constraint"

if __name__ == "__main__":
    test_monotonic_directions()
    print("\nAll monotonic constraint tests passed successfully.")


def test_blend_ablation_report_exists():
    from pathlib import Path
    import json
    reports = sorted(Path("reports_archive").glob("exp_g_blend_ablation_*.json"))
    if not reports:
        import pytest
        pytest.skip("Run exp_g_blend_ablation.py first")
    with open(reports[-1]) as f:
        r = json.load(f)
    assert "jd_marginal_contribution" in r
    assert "yoe_marginal_contribution" in r
    # Both should be computed (positive or not)
    assert isinstance(r["jd_marginal_contribution"],  float)
    assert isinstance(r["yoe_marginal_contribution"], float)


def test_no_feature_dominates_old_gbm():
    """No single feature in the old GBM should have > 50% of importance."""
    from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES
    from pathlib import Path
    model = GBMLambdaRankModel()
    model.load("ranking_lab/models/gbm_lambdarank.txt")
    imps = model.model.feature_importance()
    total = sum(imps)
    top_imp = max(imps)
    assert top_imp / total < 0.50, (
        f"Top feature has {top_imp/total:.1%} of GBM importance — too dominant"
    )
