import pytest
from retrieval_lab.fusion.rrf import reciprocal_rank_fusion

def test_rrf_tie_breaking():
    # Construct two lists of candidates with exactly zero overlap
    # To test if tie-breaking works, we give list2 slightly higher normalized raw scores 
    # to see if list2 wins the ties when list1 is processed first
    list1 = [
        {"candidate_id": "A0", "score": 10.0},
        {"candidate_id": "A1", "score": 5.0}
    ]
    # list1 raw normalized: A0=1.0, A1=0.0
    
    list2 = [
        {"candidate_id": "B0", "score": 0.99},
        {"candidate_id": "B1", "score": 0.1}
    ]
    # list2 raw normalized: B0=1.0, B1=0.0
    
    # If tie-breaker relies on order, list1 items (A0, A1) would beat list2 items (B0, B1).
    # But wait, A0 and B0 have the EXACT same normalized score (1.0). So they will tie.
    # To ensure one definitively beats the other based on normalized score, let's change list2:
    list2_adjusted = [
        {"candidate_id": "B0", "score": 1.0},  # max
        {"candidate_id": "B1", "score": 0.6},  # 0.4 / 0.8 = 0.5 normalized
        {"candidate_id": "B2", "score": 0.2}   # min
    ]
    list1_adjusted = [
        {"candidate_id": "A0", "score": 10.0}, # max
        {"candidate_id": "A1", "score": 2.8},  # 1.8 / 9.0 = 0.2 normalized
        {"candidate_id": "A2", "score": 1.0}   # min
    ]
    
    # At rank 0: A0 (norm=1.0) vs B0 (norm=1.0).
    # At rank 1: A1 (norm=0.2) vs B1 (norm=0.5). Base RRF equal. B1 should win tie-breaker.
    
    results = reciprocal_rank_fusion([list1_adjusted, list2_adjusted])
    
    # Expected order:
    # 1. A0 or B0 (tie, A0 first due to stability)
    # 2. B0 or A0
    # 3. B1 (because normalized tie-breaker 0.25 > 0.11)
    # 4. A1
    
    ranked_ids = [res["candidate_id"] for res in results]
    
    # Verify B1 ranks above A1 despite list2 being passed second
    assert ranked_ids.index("B1") < ranked_ids.index("A1"), f"RRF tie-breaker failed to prioritize higher normalized score. Order: {ranked_ids}"

def test_rrf_identical_score_handling():
    # If all candidates in a list have the exact same score, range_score is 0.
    # The normalization logic uses range_score = 1.0 in this case.
    list1 = [
        {"candidate_id": "A0", "score": 5.0},
        {"candidate_id": "A1", "score": 5.0}
    ]
    results = reciprocal_rank_fusion([list1])
    # Should not throw divide by zero
    assert len(results) == 2
