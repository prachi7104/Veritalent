from fastapi import APIRouter
from backend.api.schemas.responses import HealthResponse
from backend.config import MODEL_VERSION
from backend.data_access.feature_store_repository import get_count, get_freshness
from backend.services.explainability_service import get_narratives_count
from backend.services.retrieval_service import is_loaded as is_retrieval_loaded
from backend.services.ranking_service import is_loaded as is_ranking_loaded

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    status = "ok"
    if get_count() == 0 or not is_ranking_loaded():
        status = "degraded"
        
    return HealthResponse(
        status=status,
        model_version=MODEL_VERSION,
        feature_store_rows=get_count(),
        feature_store_freshness=get_freshness(),
        narratives_cached=get_narratives_count(),
        dense_index_loaded=is_retrieval_loaded(),
        retrieval_model="BAAI/bge-small-en-v1.5"
    )
