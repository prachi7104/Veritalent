import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

FEATURE_STORE_PATH = "feature_lab/store/feature_store.jsonl"
_store: dict[str, dict] = {}
_freshness = "Unknown"

def load():
    global _store, _freshness
    logger.info(f"Loading feature store from {FEATURE_STORE_PATH}...")
    try:
        if not os.path.exists(FEATURE_STORE_PATH):
            logger.error("feature_store.jsonl missing")
            raise FileNotFoundError("feature_store.jsonl missing")

        mtime = os.path.getmtime(FEATURE_STORE_PATH)
        _freshness = datetime.fromtimestamp(mtime).isoformat()

        with open(FEATURE_STORE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                _store[c["candidate_id"]] = c
        logger.info(f"Loaded {len(_store)} feature store rows.")
    except Exception as e:
        logger.error(f"Failed to load feature store: {e}")
        raise

def get_features(candidate_id: str) -> dict | None:
    return _store.get(candidate_id)

def get_features_batch(candidate_ids: list[str]) -> dict[str, dict]:
    return {cid: _store[cid] for cid in candidate_ids if cid in _store}

def get_count() -> int:
    return len(_store)

def get_freshness() -> str:
    return _freshness
