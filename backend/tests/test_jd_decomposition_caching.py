import pytest
import os
import shutil
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

@pytest.fixture(autouse=True)
def clean_cache():
    cache_dir = "backend/cache/jd_decompositions"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    yield
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def test_jd_decomposition_caching():
    jd_text = "We need an AI engineer who knows PyTorch. " * 5
    
    with patch('backend.services.jd_decomposition_service._try_llm_decomposition') as mock_try:
        mock_try.return_value = {"must_haves": ["PyTorch"], "nice_to_haves": [], "hard_exclusions": [], "experience_band": "Unknown", "logistics": {}}
        
        with TestClient(app) as client:
            response1 = client.request("POST", "/search", json={"jd_text": jd_text, "top_k": 10})
            assert mock_try.call_count == 1
            
            response2 = client.request("POST", "/search", json={"jd_text": jd_text, "top_k": 10})
            assert mock_try.call_count == 1
