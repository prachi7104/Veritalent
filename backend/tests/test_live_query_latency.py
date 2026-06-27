import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

def test_live_query_latency():
    jd_text = "Expert python engineer for ranking systems. " * 5
    
    with patch('backend.services.jd_decomposition_service._try_llm_decomposition') as mock_try:
        mock_try.return_value = {"must_haves": ["python"], "nice_to_haves": [], "hard_exclusions": [], "experience_band": "Unknown", "logistics": {}}
        
        with TestClient(app) as client:
            client.request("POST", "/search", json={"jd_text": jd_text, "top_k": 20})
            
            start_time = time.time()
            response = client.request("POST", "/search", json={"jd_text": jd_text, "top_k": 20})
            duration_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert duration_ms < 800.0, f"Latency {duration_ms}ms exceeded 800ms budget"
