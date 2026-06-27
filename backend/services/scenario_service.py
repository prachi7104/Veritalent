from cachetools import TTLCache
from backend.config import SCENARIO_FEATURE_GROUPS, SCENARIO_DEFAULT_WEIGHT

_session_store: TTLCache = TTLCache(maxsize=500, ttl=3600)

def create_session(session_id: str, candidate_ids: list[str], features: dict[str, dict]) -> None:
    _session_store[session_id] = {
        "candidate_ids": candidate_ids,
        "features": features,
        "original_ranks": {cid: i+1 for i, cid in enumerate(candidate_ids)},
    }

def get_session(session_id: str) -> dict | None:
    return _session_store.get(session_id)

def scenario_rerank(session_id: str, weight_overrides: dict) -> list[dict]:
    session = _session_store.get(session_id)
    if session is None:
        raise KeyError("session_expired")

    candidate_ids = session["candidate_ids"]
    features = session["features"]
    original_ranks = session["original_ranks"]

    weights = {g: weight_overrides.get(g, SCENARIO_DEFAULT_WEIGHT) for g in SCENARIO_FEATURE_GROUPS}
    total_weight = sum(weights.values()) or 1.0

    all_values: dict[str, list[float]] = {f: [] for group in SCENARIO_FEATURE_GROUPS.values() for f in group}
    for f in all_values:
        all_values[f] = [float(features.get(cid, {}).get(f, 0.0) or 0.0) for cid in candidate_ids]

    def normalize(f, val):
        lo, hi = min(all_values[f]), max(all_values[f])
        if hi == lo:
            return 0.5
        return (val - lo) / (hi - lo)

    scores = {}
    for cid in candidate_ids:
        feat = features.get(cid, {})
        score = 0.0
        for group, f_list in SCENARIO_FEATURE_GROUPS.items():
            if not f_list:
                continue
            group_vals = []
            for f in f_list:
                raw = float(feat.get(f, 0.0) or 0.0)
                norm = normalize(f, raw)
                if f == "trust_score":
                    norm = 1.0 - norm
                group_vals.append(norm)
            group_mean = sum(group_vals) / len(group_vals)
            score += (weights[group] / total_weight) * group_mean
        scores[cid] = score

    re_ranked_ids = sorted(candidate_ids, key=lambda c: scores[c], reverse=True)

    result = []
    for new_rank, cid in enumerate(re_ranked_ids, start=1):
        orig = original_ranks.get(cid, new_rank)
        result.append({
            "candidate_id": cid,
            "new_rank": new_rank,
            "original_rank": orig,
            "rank_delta": orig - new_rank,
            "scenario_score": round(scores[cid], 6),
        })
    return result
