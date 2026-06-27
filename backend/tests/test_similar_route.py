import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_similar_route_is_removed():
    response = client.get("/candidate/some_id/similar")
    assert response.status_code == 404
    assert response.json()["detail"] == "Similar candidates feature removed"
