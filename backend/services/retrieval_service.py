import os
import logging
from retrieval_lab.indexing.dense_index import DenseIndex
from backend.data_access.feature_store_repository import get_features_batch

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

def retrieve(jd_decomposition: dict, top_k: int = 200) -> list[str]:
    must_haves = " ".join(jd_decomposition.get("must_haves", []))
    nice_to_haves = " ".join(jd_decomposition.get("nice_to_haves", []))
    query = f"{must_haves} {nice_to_haves}".strip()

    if not query:
        query = jd_decomposition.get("experience_band", "")

    if not query:
        return _fallback_retrieval(top_k)

    try:
        results = _dense_index.search(query, top_k=top_k)
        candidate_ids = [r["candidate_id"] for r in results]
        
        if not candidate_ids:
            return _fallback_retrieval(top_k)
            
        return candidate_ids
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}. Using fallback.")
        return _fallback_retrieval(top_k)

def _fallback_retrieval(top_k: int) -> list[str]:
    logger.warning("Using fallback retrieval by skill_depth.")
    candidate_ids = _dense_index.corpus_ids
    
    features = get_features_batch(candidate_ids)
    
    scored = []
    for cid in candidate_ids:
        feat = features.get(cid)
        depth = float(feat.get("skill_depth", 0.0) if feat else 0.0)
        scored.append((cid, depth))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in scored[:top_k]]
