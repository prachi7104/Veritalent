import pytest


def test_build_candidate_text_includes_key_fields():
    from retrieval_lab.rerankers.cross_encoder_reranker import build_candidate_text
    candidate = {
        "profile": {"headline": "Senior ML Engineer", "summary": "Building search infra.",
                    "years_of_experience": 7},
        "career_history": [{"title": "ML Engineer", "company": "Flipkart",
                             "duration_months": 36, "is_current": True}],
        "skills": [{"name": "Elasticsearch"}, {"name": "Python"}]
    }
    text = build_candidate_text(candidate)
    assert "Senior ML Engineer" in text
    assert "Elasticsearch" in text


def test_cross_encoder_reranker_preserves_count():
    try:
        from retrieval_lab.rerankers.cross_encoder_reranker import CrossEncoderReranker
        r = CrossEncoderReranker()
        if not r._available:
            pytest.skip("sentence-transformers not installed")
    except Exception:
        pytest.skip("CE not available")
    candidates = [{"profile": {"headline": f"Eng {i}"}, "career_history": [],
                   "skills": []} for i in range(5)]
    ids = [f"CAND_{i}" for i in range(5)]
    reranked, scores, lat = r.rerank("ML Engineer role", candidates, ids, top_n=5)
    assert len(reranked) == 5
    assert lat > 0


def test_latency_under_5s_for_20_candidates():
    try:
        from retrieval_lab.rerankers.cross_encoder_reranker import CrossEncoderReranker
        r = CrossEncoderReranker()
        if not r._available:
            pytest.skip("sentence-transformers not installed")
    except Exception:
        pytest.skip("CE not available")
    cands = [{"profile": {"headline": f"ML Engineer {i}",
                          "summary": "Building retrieval systems at scale."},
              "career_history": [{"title": "Engineer", "company": f"Co{i}",
                                  "duration_months": 24}],
              "skills": [{"name": "Elasticsearch"}]}
             for i in range(20)]
    ids = [f"CAND_{i:04d}" for i in range(20)]
    _, _, lat = r.rerank("Senior AI/ML Engineer search retrieval", cands, ids, top_n=20)
    assert lat < 5000, f"20-candidate CE rerank took {lat:.0f}ms — too slow"
