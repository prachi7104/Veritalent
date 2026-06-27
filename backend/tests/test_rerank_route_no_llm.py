import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app
import hashlib

def test_rerank_route_no_llm():
    jd_text = "Looking for a Machine Learning engineer with deep learning skills"
    with patch('backend.services.jd_decomposition_service.httpx.Client.post') as mock_post:
        with TestClient(app) as client:
            response = client.request("POST", "/rerank", json={
                "session_id": hashlib.sha256(jd_text.encode()).hexdigest(),
                "updated_jd_text": jd_text
            })
            assert not mock_post.called, "Rerank route MUST NOT call the LLM"
