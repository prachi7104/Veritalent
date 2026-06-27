from fastapi import APIRouter, HTTPException
from backend.api.schemas.requests import SearchRequest
from backend.api.schemas.responses import SearchResponse
from backend.pipelines.live_query_pipeline import execute_live_query

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        return execute_live_query(
            jd_text=request.jd_text,
            top_k=request.top_k,
            include_trust=request.include_trust,
            allow_llm=True
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
