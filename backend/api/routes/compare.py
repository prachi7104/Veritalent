from fastapi import APIRouter, HTTPException
from backend.api.schemas.requests import CompareRequest
from backend.api.schemas.responses import CompareResponse, ComparisonMatrix
from backend.api.routes.candidate import get_candidate_detail

router = APIRouter()

@router.post("/compare", response_model=CompareResponse)
def compare_candidates(request: CompareRequest):
    if not (2 <= len(request.candidate_ids) <= 4):
        raise HTTPException(status_code=422, detail="Compare supports 2–4 candidates maximum.")
        
    candidates = []
    for cid in request.candidate_ids:
        try:
            cand = get_candidate_detail(cid)
            candidates.append(cand)
        except HTTPException:
            pass
            
    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates found")
        
    top_cand = candidates[0]
    
    features_to_compare = [
        "skill_depth", "skill_breadth", "tenure_stability", 
        "promotion_velocity", "activity_quality_composite", "trust_score"
    ]
    
    values = {}
    deltas = {}
    
    for feat in features_to_compare:
        values[feat] = {}
        deltas[feat] = {}
        top_val = float(getattr(top_cand.features, feat, 0.0) or 0.0)
        
        for cand in candidates:
            val = float(getattr(cand.features, feat, 0.0) or 0.0)
            values[feat][cand.candidate_id] = val
            deltas[feat][cand.candidate_id] = val - top_val
            
    matrix = ComparisonMatrix(
        features=features_to_compare,
        values=values,
        deltas=deltas
    )
    
    return CompareResponse(
        candidates=candidates,
        comparison_matrix=matrix
    )
