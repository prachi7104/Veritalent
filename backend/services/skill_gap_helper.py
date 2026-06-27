from backend.config import DEEP_IR_SKILLS

def compute_skill_gap(candidate_raw: dict, jd_decomposition: dict,
                      rank: int, ranked_candidates_features: list[dict]) -> dict:
    """
    candidate_raw: full candidate dict from candidate_repository
    jd_decomposition: structured output from jd_decomposition_service (as dict)
    rank: this candidate's GBM rank in the current shortlist (1-indexed)
    ranked_candidates_features: feature dicts for all candidates in the
                                 current shortlist, in rank order
    """
    candidate_skill_names = {s["name"] for s in candidate_raw.get("skills", [])}

    jd_required = set(jd_decomposition.get("must_haves", []))
    relevant_required = (jd_required & DEEP_IR_SKILLS) if jd_required else DEEP_IR_SKILLS

    matched = sorted(candidate_skill_names & relevant_required)
    missing = sorted(relevant_required - candidate_skill_names)

    gap_text = None
    if rank > 3 and len(ranked_candidates_features) >= 3:
        target = ranked_candidates_features[2]   # rank-3 candidate features
        current = next((f for f in ranked_candidates_features
                        if f["candidate_id"] == candidate_raw["candidate_id"]), None)
        if current and target:
            depth_gap = target["skill_depth"] - current["skill_depth"]
            tenure_gap = target["tenure_stability"] - current["tenure_stability"]
            if depth_gap > 0 or tenure_gap > 0:
                parts = []
                if depth_gap > 20:
                    approx_skills = max(1, round(depth_gap / 36))  # ~36mo per skill
                    parts.append(f"{approx_skills} more deep-IR skill"
                                 f"{'s' if approx_skills > 1 else ''}")
                if tenure_gap > 6:
                    parts.append(f"{round(tenure_gap)} months more tenure stability")
                if parts:
                    gap_text = (f"{' and '.join(parts)} would move this candidate "
                                f"closer to rank #3.")

    return {
        "missing_deep_ir_skills": missing,
        "matched_deep_ir_skills": matched,
        "gap_to_next_tier": gap_text,
    }
