from fastapi import APIRouter, HTTPException
from backend.api.schemas.responses import CandidateDetailResponse, ProfileDetail, CareerEntry
from backend.data_access.candidate_repository import get as get_candidate
from backend.services.feature_service import get_features
from backend.services.trust_service import get_trust_breakdown
from backend.services.explainability_service import get_shap_attribution, get_narrative

router = APIRouter()

@router.get("/candidate/{candidate_id}", response_model=CandidateDetailResponse)
def get_candidate_detail(candidate_id: str):
    raw = get_candidate(candidate_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    feat = get_features(candidate_id) or {}
    
    trust_breakdown = get_trust_breakdown(raw)
    shap_attr = get_shap_attribution(feat)
    
    narrative, narrative_is_llm, _ = get_narrative(candidate_id, shap_attr.get("top_features", []))
    
    career_history = []
    for entry in raw.get("career_timeline", []):
        career_history.append(CareerEntry(
            company=entry.get("company", "Unknown") or "Unknown",
            title=entry.get("title", "Unknown") or "Unknown",
            start_date=entry.get("start_date", "Unknown") or "Unknown",
            end_date=entry.get("end_date"),
            duration_months=int(entry.get("duration_months", 0) or 0),
            is_current=bool(entry.get("is_current", False)),
            industry=entry.get("industry", "Unknown") or "Unknown"
        ))
        
    profile = ProfileDetail(
        current_title=raw.get("profile", {}).get("current_title", "Unknown") or "Unknown",
        current_company=raw.get("profile", {}).get("current_company", "Unknown") or "Unknown",
        years_of_experience=float(raw.get("profile", {}).get("years_of_experience", 0.0) or 0.0),
        headline=raw.get("profile", {}).get("headline", "Unknown") or "Unknown",
        summary=raw.get("profile", {}).get("summary", "Unknown") or "Unknown",
        location=raw.get("profile", {}).get("location", "Unknown") or "Unknown",
        country=raw.get("profile", {}).get("country", "Unknown") or "Unknown",
        career_history=career_history
    )
    
    return CandidateDetailResponse(
        candidate_id=candidate_id,
        profile=profile,
        features=feat,
        trust_breakdown=trust_breakdown,
        shap_attribution=shap_attr,
        narrative=narrative,
        narrative_is_llm=narrative_is_llm,
        fingerprint_holder=raw.get("fingerprint_holder", False)
    )

@router.get("/candidate/{candidate_id}/similar")
def get_similar_candidates(candidate_id: str):
    raise HTTPException(status_code=404, detail="Similar candidates feature removed")
