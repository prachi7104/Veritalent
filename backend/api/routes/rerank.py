from fastapi import APIRouter, HTTPException
from backend.api.schemas.requests import RerankRequest
from backend.api.schemas.responses import SearchResponse
from backend.pipelines.live_query_pipeline import execute_live_query

router = APIRouter()

@router.post("/rerank", response_model=SearchResponse)
def rerank(request: RerankRequest):
    try:
        return execute_live_query(
            jd_text=request.updated_jd_text,
            top_k=20,
            include_trust=True,
            allow_llm=False
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
