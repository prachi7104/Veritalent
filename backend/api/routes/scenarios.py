from fastapi import APIRouter, HTTPException
from backend.api.schemas.requests import ScenarioRerankRequest
from backend.api.schemas.responses import ScenarioRerankResponse, ScenarioCandidate
from backend.services.scenario_service import scenario_rerank

router = APIRouter()

@router.post("/scenarios/rerank", response_model=ScenarioRerankResponse)
def run_scenario(request: ScenarioRerankRequest):
    try:
        results = scenario_rerank(request.session_id, request.weight_overrides.model_dump(exclude_unset=True))
        
        candidates = [ScenarioCandidate(**r) for r in results]
        
        from backend.config import SCENARIO_DEFAULT_WEIGHT, SCENARIO_FEATURE_GROUPS
        applied = {g: SCENARIO_DEFAULT_WEIGHT for g in SCENARIO_FEATURE_GROUPS}
        for k, v in request.weight_overrides.model_dump(exclude_unset=True).items():
            if k in applied:
                applied[k] = v
            
        return ScenarioRerankResponse(
            re_ranked=candidates,
            weight_applied=applied,
            note="Scenario scores use the linear baseline model with adjusted weights. Results differ from the primary GBM ranking."
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Session expired or not found. Please re-submit your search.")
