import pytest
from retrieval_lab.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    ranked_ids = ["c1", "c2", "c3", "c4"]
    gold = {"c1": 3, "c2": 1, "c3": 2, "c4": 0}
    
    # top 2: c1 (rel), c2 (not rel) -> P@2 = 0.5
    assert precision_at_k(ranked_ids, gold, 2, threshold=2) == 0.5
    
    # top 4: c1 (rel), c2 (not rel), c3 (rel), c4 (not rel) -> P@4 = 0.5
    assert precision_at_k(ranked_ids, gold, 4, threshold=2) == 0.5

def test_recall_at_k():
    ranked_ids = ["c1", "c2", "c3", "c4"]
    gold = {"c1": 3, "c2": 1, "c3": 2, "c4": 0, "c5": 3} # total 3 rel (c1, c3, c5)
    
    # top 2: c1 (rel) -> R@2 = 1/3
    assert abs(recall_at_k(ranked_ids, gold, 2, threshold=2) - 1/3) < 1e-6
    
    # top 4: c1, c3 (rel) -> R@4 = 2/3
    assert abs(recall_at_k(ranked_ids, gold, 4, threshold=2) - 2/3) < 1e-6

def test_ndcg_at_k():
    ranked_ids = ["c1", "c2", "c3"]
    gold = {"c1": 3, "c2": 2, "c3": 0, "c4": 3}
    
    # DCG@2: 
    # c1 (3) -> (8-1)/log2(2) = 7
    # c2 (2) -> (4-1)/log2(3) = 3/1.58 = 1.89
    # DCG = 8.89
    # IDCG@2 (ideal scores 3, 3):
    # (8-1)/1 + (8-1)/1.58 = 7 + 4.41 = 11.41
    # NDCG@2 = 8.89 / 11.41 ≈ 0.779
    
    ndcg = ndcg_at_k(ranked_ids, gold, 2)
    assert 0.77 < ndcg < 0.78
