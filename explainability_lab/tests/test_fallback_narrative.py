import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from explainability_lab.narrative.fallback_narrative import generate_fallback_narrative

def test_fallback_contains_no_raw_feature_names():
    """Fallback narrative must not expose raw feature names."""
    context = {
        "rank": 5,
        "current_title": "ML Engineer",
        "current_company": "Flipkart",
        "company_type": "primarily product company",
        "yoe": "7",
        "yoe_band_label": "7 years — target band ✓",
        "jd_skills": "Elasticsearch, PyTorch",
        "shap_summary": "JD-aligned skill strength (+4.2, primary driver)",
        "trust_label": "clean profile",
        "notice_period": "30 days",
    }
    text = generate_fallback_narrative(context)
    raw_features = ["skill_mastery_triangulation", "skill_depth",
                    "activity_quality_composite"]
    for feat in raw_features:
        assert feat not in text, f"Fallback contains raw feature name: {feat}"


def test_shap_formatter_translates_feature_names():
    """SHAP formatter must translate raw names to English."""
    from explainability_lab.narrative.shap_formatter import format_shap_for_narrative
    raw_shap = [
        {"feature": "skill_mastery_triangulation", "shap_value": 4.5},
        {"feature": "logistics_fit_score", "shap_value": 1.2},
        {"feature": "trust_score", "shap_value": -0.3},
    ]
    result = format_shap_for_narrative(raw_shap)
    assert "skill_mastery_triangulation" not in result
    assert "verified skill depth" in result
    assert "location and availability fit" in result
    assert "concern" in result  # trust negative should be labeled concern


def test_blend_annotation_high_jd_score():
    """Candidate with jd_skill_score well above mean gets 'strong JD alignment' label."""
    from explainability_lab.narrative.candidate_context import build_blend_annotation
    features = {"jd_skill_score": 150.0, "yoe_band_fit": 1.0}
    ann = build_blend_annotation(features, pool_jd_mean=50.0, pool_yoe_mean=0.64)
    assert "strong JD alignment" in ann["jd_skill_label"]


def test_blend_annotation_target_yoe_band():
    """Candidate in 5-9 YOE band gets target band label."""
    from explainability_lab.narrative.candidate_context import build_blend_annotation
    features = {"jd_skill_score": 60.0, "yoe_band_fit": 1.0}
    ann = build_blend_annotation(features, pool_jd_mean=50.0, pool_yoe_mean=0.64)
    assert "target band" in ann["yoe_band_label_text"]


def test_narrative_cache_references_jd_domain():
    """At least 30% of narratives should reference IR/search domain."""
    from pathlib import Path
    import json
    p = Path("submission/narratives_cache.json")
    if not p.exists():
        import pytest; pytest.skip("Cache not built yet")
    with open(p) as f:
        cache = json.load(f)
    jd_refs = sum(
        1 for entry in cache.values()
        if any(ph in entry["narrative"].lower() for ph in
               ["search","retrieval","ranking","embedding","vector","information retrieval"])
    )
    assert jd_refs >= 30, f"Only {jd_refs}/100 narratives reference JD domain"


def test_shap_values_differ_across_candidates():
    """
    After SHAP fix, SHAP values should differ across candidates.
    Identical SHAP = explainer receiving wrong feature vectors.
    """
    import pytest, json, numpy as np
    from pathlib import Path
    from ranking_lab.models.gbm_lambdarank import GBMLambdaRankModel
    from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

    store_path = Path("feature_lab/store/feature_store.jsonl")
    sub_path   = Path("submission/submission.csv")
    if not store_path.exists() or not sub_path.exists():
        pytest.skip("Feature store or submission not available")

    import shap, pandas as pd
    model = GBMLambdaRankModel()
    model.load("ranking_lab/models/gbm_lambdarank.txt")
    explainer = shap.TreeExplainer(model.model)
    num_feats = model.model.num_feature()
    feats_10 = TRAINING_FEATURES[:num_feats]

    store = {}
    with open(store_path) as f:
        for line in f:
            row = json.loads(line); store[row["candidate_id"]] = row

    sub = pd.read_csv(sub_path)
    all_svs = []
    for rank in [1, 5, 10, 25, 50]:
        row = sub[sub["rank"] == rank].iloc[0]
        feats = store.get(row["candidate_id"], {})
        X = np.array([float(feats.get(f, 0) or 0) for f in feats_10])
        sv = explainer.shap_values(X.reshape(1, -1))[0]
        all_svs.append(sv)

    # Verify that the SHAP vectors are candidate-specific and not identical
    for i in range(len(all_svs)):
        for j in range(i+1, len(all_svs)):
            assert not np.allclose(all_svs[i], all_svs[j]), f"SHAP vectors for ranks are identical!"


def test_top50_narratives_not_all_same_opener():
    """
    After Fix C (rank-aware prompt), ranks 1-5 should NOT all open
    with 'exceptional breadth of technical skills'.
    """
    import json
    from pathlib import Path
    p = Path("submission/narratives_cache.json")
    if not p.exists():
        import pytest; pytest.skip("Cache not built")
    with open(p) as f:
        cache = json.load(f)
    openers = [cache[str(r)]["narrative"][:40] for r in range(1, 6)]
    unique = len(set(openers))
    assert unique >= 3, (
        f"Only {unique}/5 unique openings in ranks 1-5: {openers}. "
        "Top candidates are still getting templated openings."
    )

