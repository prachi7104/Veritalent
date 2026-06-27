import json
import logging
from backend.config import FINGERPRINT_HOLDERS

logger = logging.getLogger(__name__)

CANDIDATES_PATH = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
_store: dict[str, dict] = {}

def load():
    global _store
    logger.info(f"Loading candidates from {CANDIDATES_PATH}...")
    try:
        with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                c_id = c["candidate_id"]
                c["fingerprint_holder"] = c_id in FINGERPRINT_HOLDERS
                _store[c_id] = c
        logger.info(f"Loaded {len(_store)} candidates into memory.")
    except Exception as e:
        logger.error(f"Failed to load candidates: {e}")
        raise

def get(candidate_id: str) -> dict | None:
    return _store.get(candidate_id)
