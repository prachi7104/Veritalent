import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

def test_rerank_top_k():
    jd_text = "Senior Python Developer with expert backend skills. " * 5
    updated_jd_text = "Expert Python Developer with expert backend skills and API experience. " * 5
    
    with patch('backend.services.jd_decomposition_service._try_llm_decomposition') as mock_try:
        mock_try.return_value = {
            "must_haves": ["Python"],
            "nice_to_haves": [],
            "hard_exclusions": [],
            "experience_band": "Unknown",
            "logistics": {}
        }
        
        with TestClient(app) as client:
            # 1. POST /search to create a session
            response = client.post("/search", json={"jd_text": jd_text, "top_k": 50})
            assert response.status_code == 200
            search_data = response.json()
            session_id = search_data["session_id"]
            
            # 2. POST /rerank with no top_k (should default to 100)
            rerank_resp = client.post("/rerank", json={
                "session_id": session_id,
                "updated_jd_text": updated_jd_text
            })
            assert rerank_resp.status_code == 200
            rerank_data = rerank_resp.json()
            cands = rerank_data["candidates"]
            assert len(cands) > 20
            assert len(cands) <= 100
            
            # 3. POST /rerank with invalid top_k (150) should be rejected (422)
            invalid_resp = client.post("/rerank", json={
                "session_id": session_id,
                "updated_jd_text": updated_jd_text,
                "top_k": 150
            })
            assert invalid_resp.status_code == 422
