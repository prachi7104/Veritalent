import os
import logging
import re
from retrieval_lab.indexing.dense_index import DenseIndex
from backend.data_access.feature_store_repository import get_features_batch
from backend.data_access.candidate_repository import get as get_candidate

logger = logging.getLogger(__name__)

INDEX_PATH = "retrieval_lab/cache/dense.npz"
_dense_index = None

def load():
    global _dense_index
    logger.info(f"Loading dense index from {INDEX_PATH}...")
    _dense_index = DenseIndex()
    if os.path.exists(INDEX_PATH):
        _dense_index.load(INDEX_PATH)
        logger.info(f"Dense index loaded with {len(_dense_index.corpus_ids)} candidates.")
    else:
        logger.error(f"Dense index not found at {INDEX_PATH}")
        raise FileNotFoundError(f"Dense index not found at {INDEX_PATH}")

def is_loaded() -> bool:
    return _dense_index is not None and _dense_index.embeddings is not None

def _apply_hard_exclusions(candidate_ids: list[str], hard_exclusions: list[str], top_k: int) -> list[str]:
    if not hard_exclusions:
        return candidate_ids[:top_k]

    excluded_patterns = []
    for ex in hard_exclusions:
        # Match exclusion terms using case-insensitive whole-word matching
        excluded_patterns.append(re.compile(rf"\b{re.escape(ex)}\b", re.IGNORECASE))

    filtered_ids = []
    for cid in candidate_ids:
        cand_data = get_candidate(cid)
        if cand_data:
            summary = cand_data.get("profile", {}).get("summary") or ""
            headline = cand_data.get("profile", {}).get("headline") or ""
            skills = " ".join([s.get("name", "") for s in cand_data.get("skills", [])])
            text_to_check = f"{summary} {headline} {skills}"
            
            if any(p.search(text_to_check) for p in excluded_patterns):
                continue
                
        filtered_ids.append(cid)
        if len(filtered_ids) >= top_k:
            break
            
    return filtered_ids

def retrieve(jd_decomposition: dict, top_k: int = 200) -> list[str]:
    hard_exclusions = jd_decomposition.get("hard_exclusions", [])
    
    must_haves = " ".join(jd_decomposition.get("must_haves", []))
    nice_to_haves = " ".join(jd_decomposition.get("nice_to_haves", []))
    query = f"{must_haves} {nice_to_haves}".strip()

    if not query:
        query = jd_decomposition.get("experience_band", "")

    if not query:
        return _fallback_retrieval(top_k, hard_exclusions)

    try:
        results = _dense_index.search(query, top_k=None)
        candidate_ids = [r["candidate_id"] for r in results]
        
        if not candidate_ids:
            return _fallback_retrieval(top_k, hard_exclusions)
            
        return _apply_hard_exclusions(candidate_ids, hard_exclusions, top_k)
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}. Using fallback.")
        return _fallback_retrieval(top_k, hard_exclusions)

def _fallback_retrieval(top_k: int, hard_exclusions: list[str] = None) -> list[str]:
    if hard_exclusions is None:
        hard_exclusions = []
        
    logger.warning("Using fallback retrieval by skill_depth.")
    candidate_ids = _dense_index.corpus_ids
    
    features = get_features_batch(candidate_ids)
    
    scored = []
    for cid in candidate_ids:
        feat = features.get(cid)
        depth = float(feat.get("skill_depth", 0.0) if feat else 0.0)
        scored.append((cid, depth))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    sorted_candidate_ids = [c[0] for c in scored]
    
    return _apply_hard_exclusions(sorted_candidate_ids, hard_exclusions, top_k)
