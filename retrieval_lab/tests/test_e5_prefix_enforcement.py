import pytest
from retrieval_lab.indexing.shootout_index import ShootoutIndex

def test_e5_prefix_enforcement():
    # Verify that the ShootoutIndex correctly prepends prefixes
    idx = ShootoutIndex(
        model_name="intfloat/e5-small-v2", 
        query_prefix="query: ", 
        passage_prefix="passage: "
    )
    
    cand = {
        "candidate_id": "test_1",
        "profile": {
            "summary": "This is a summary.",
            "headline": "This is a headline."
        },
        "skills": [{"name": "Python"}, {"name": "Machine Learning"}]
    }
    
    extracted = idx._extract_text(cand)
    assert extracted.startswith("passage: "), f"Missing passage prefix: {extracted}"
    assert "This is a summary." in extracted
    assert "Skills: Python, Machine Learning" in extracted
    
    # We can't easily intercept the query prefix without mocking SentenceTransformer, 
    # but we can verify the class property is set
    assert idx.query_prefix == "query: "
