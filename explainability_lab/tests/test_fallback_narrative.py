import pytest
import os
import sys
import socket

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from explainability_lab.narrative.fallback_narrative import generate_fallback

def block_network():
    # Monkeypatch socket to ensure no network calls are made
    def guard(*args, **kwargs):
        raise RuntimeError("Network call blocked during fallback narrative test!")
    socket.socket = guard

def test_fallback_narrative_no_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", block_network)
    
    shap_summary = [
        {"feature": "skill_depth", "raw_value": 4.0, "shap_value": 1.2},
        {"feature": "trust_score", "raw_value": 0.5, "shap_value": -0.8}
    ]
    
    narrative = generate_fallback(shap_summary)
    
    assert "skill_depth" in narrative
    assert "trust_score" in narrative
    assert "impact: +" in narrative
    assert "impact: -" in narrative
