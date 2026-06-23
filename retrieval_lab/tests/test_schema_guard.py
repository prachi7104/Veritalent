import pytest
import math
from retrieval_lab.data.schema_guard import enforce_schema, SchemaViolationError

def test_enforce_schema_sentinel():
    cand = {
        "redrob_signals": {
            "github_activity_score": -1,
            "offer_acceptance_rate": -1
        }
    }
    
    guarded = enforce_schema(cand)
    assert math.isnan(guarded["redrob_signals"]["github_activity_score"])
    assert math.isnan(guarded["redrob_signals"]["offer_acceptance_rate"])

def test_enforce_schema_education():
    cand = {
        "education": [
            {
                "degree": "BSc",
                "start_year": 2010,
                "end_year": 2014
            }
        ]
    }
    
    guarded = enforce_schema(cand)
    edu = guarded["education"][0]
    
    # Allowed access
    assert edu["degree"] == "BSc"
    assert edu.get("degree") == "BSc"
    
    # Blocked access
    with pytest.raises(SchemaViolationError):
        _ = edu["start_year"]
        
    with pytest.raises(SchemaViolationError):
        _ = edu.get("end_year")
