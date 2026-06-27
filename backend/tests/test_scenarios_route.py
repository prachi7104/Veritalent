import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.scenario_service import create_session

def test_scenarios_route():
    session_id = "test_scenario_session"
    create_session(session_id, ["cand1", "cand2"], {
        "cand1": {"skill_depth": 0.9, "trust_score": 0.9}, # high skill, high risk
        "cand2": {"skill_depth": 0.1, "trust_score": 0.1}  # low skill, low risk
    })
    
    with TestClient(app) as client:
        response = client.post("/scenarios/rerank", json={
            "session_id": session_id,
            "weight_overrides": {"skills": 100.0, "trust": 0.0}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["re_ranked"][0]["candidate_id"] == "cand1"
        
        response2 = client.post("/scenarios/rerank", json={
            "session_id": session_id,
            "weight_overrides": {"skills": 0.0, "trust": 100.0}
        })
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["re_ranked"][0]["candidate_id"] == "cand2"
