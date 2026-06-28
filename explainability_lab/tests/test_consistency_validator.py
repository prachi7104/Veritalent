
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from explainability_lab.narrative.consistency_validator import validate_consistency

def test_consistency_validator_pass():
    narrative = 'This is a long enough narrative to pass the new consistency validator length check of twenty words.' * 3
    assert validate_consistency(narrative, []) == True

