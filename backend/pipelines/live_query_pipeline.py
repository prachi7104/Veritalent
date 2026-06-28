import hashlib
import logging
from backend.services.jd_decomposition_service import get_decomposition, get_decomposition_no_llm
from backend.services.retrieval_service import retrieve
from backend.services.feature_service import get_features_batch
from backend.services.ranking_service import score_batch
from backend.services.scenario_service import create_session
from backend.services.explainability_service import get_shap_attribution, get_narrative
from backend.services.skill_gap_helper import compute_skill_gap
from backend.config import get_trust_level
from backend.data_access.candidate_repository import get as get_candidate
from explainability_lab.narrative.candidate_context import build_candidate_context
from backend.api.schemas.responses import (
    SearchResponse, FunnelStats, CandidateCardResponse, SkillGap, JDDecomposition
)

logger = logging.getLogger(__name__)

def execute_live_query(jd_text: str, top_k: int, include_trust: bool = True, allow_llm: bool = True) -> SearchResponse:
    if len(jd_text) < 20:
        raise ValueError("Job description is too short. Please provide at least 20 characters.")
        
    session_id = hashlib.sha256(jd_text.encode('utf-8')).hexdigest()
    
    if allow_llm:
        jd_decomp = get_decomposition(jd_text)
    else:
        jd_decomp = get_decomposition_no_llm(jd_text)
        
    candidate_ids = retrieve(jd_decomp, top_k=200)
    
    features_dict = get_features_batch(candidate_ids)
    
    ordered_cids = [cid for cid in candidate_ids if cid in features_dict]
    ordered_feats = [features_dict[cid] for cid in ordered_cids]
    
    scores = score_batch(ordered_feats)
    
    scored_cands = sorted(zip(ordered_cids, ordered_feats, scores), key=lambda x: x[2], reverse=True)
    top_cands = scored_cands[:top_k]
    
    all_cids = [c[0] for c in scored_cands]
    all_feats = {c[0]: c[1] for c in scored_cands}
    create_session(session_id, all_cids, all_feats)
    
    responses = []
    ranked_candidates_features = [c[1] for c in scored_cands]
    
    pool_jd_skill_mean = sum(float(f.get("jd_skill_score", 0)) for f in ranked_candidates_features) / max(1, len(ranked_candidates_features))
    
    for rank, (cid, feat, score) in enumerate(top_cands, start=1):
        raw = get_candidate(cid) or {}
        
        shap_attr = get_shap_attribution(feat)
        top_features = shap_attr.get("top_features", [])
        
        shap_contributions = [
            {
                "feature": f.feature_name,
                "shap_value": f.shap_contribution,
                "raw_value": f.value
            }
            for f in top_features
        ]
        
        context = build_candidate_context(
            candidate=raw,
            features=feat,
            shap_contributions=shap_contributions,
            rank=rank,
            pool_jd_skill_mean=pool_jd_skill_mean
        )
        
        narrative, narrative_is_llm, fallback_used = get_narrative(cid, context)
        
        sg = compute_skill_gap(raw, jd_decomp, rank, ranked_candidates_features)
        
        trust_score = float(feat.get("trust_score", 0.0))
        
        responses.append(CandidateCardResponse(
            candidate_id=cid,
            rank=rank,
            score=score,
            current_title=raw.get("profile", {}).get("current_title", "Unknown") or "Unknown",
            current_company=raw.get("profile", {}).get("current_company", "Unknown") or "Unknown",
            years_of_experience=float(raw.get("profile", {}).get("years_of_experience", 0.0) or 0.0),
            location=raw.get("profile", {}).get("location", "Unknown") or "Unknown",
            top_features=top_features,
            trust_score=trust_score,
            trust_level=get_trust_level(trust_score),
            fingerprint_holder=raw.get("fingerprint_holder", False),
            narrative=narrative,
            narrative_is_llm=narrative_is_llm,
            fallback_used=fallback_used,
            skill_gap=SkillGap(**sg)
        ))
        
    funnel = FunnelStats(
        total_pool=100000,
        title_relevant=31179,
        retrieved=len(candidate_ids),
        shown=len(responses)
    )
    
    jd_decomp_clean = {k: v for k, v in jd_decomp.items() if k != "fallback_used"}
    
    return SearchResponse(
        session_id=session_id,
        funnel_stats=funnel,
        candidates=responses,
        jd_decomposition=JDDecomposition(**jd_decomp_clean)
    )
