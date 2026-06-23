"""
leakage_hypothesis_test.py — Did someone plant the fingerprint skills?

Investigates the "deliberately planted evaluation artifact" hypothesis via:
  1. Textual proximity between the 13 fingerprint skill strings and the JD's exact wording.
  2. Perfect vs. near-perfect seniority alignment (a suspiciously clean pattern at n=8
     is mild evidence of synthetic construction, not organic data).
  3. Distribution anomaly: are the 13 skills grouped unnaturally tightly in their
     occurrence counts, or is the distribution what you'd expect from organic noise?

This test cannot definitively prove or disprove planting, but it can say whether
the evidence is consistent with planting, inconsistent with it, or ambiguous.

Usage:
    python -m fingerprint_lab.analysis.leakage_hypothesis_test \\
        --audit <path/to/frequency_audit_results.json>
"""
import json
import math
import argparse
from pathlib import Path

# The 13 fingerprint skill strings verbatim
FINGERPRINT_SKILLS = [
    "Search Backend",
    "Ranking Systems",
    "Text Encoders",
    "Vector Representations",
    "Content Matching",
    "Model Adaptation",
    "Information Retrieval Systems",
    "Search & Discovery",
    "Search Infrastructure",
    "Indexing Algorithms",
    "Workflow Orchestration",
    "Open-source ML libraries",
    "Document Processing",
]

# JD key phrases extracted from the target JD (Senior/Staff AI/ML Engineer — Search & Retrieval)
# These are taken verbatim or as direct paraphrases from the JD text documented in the
# master context and available in job_description.docx.
JD_KEY_PHRASES = [
    "search", "retrieval", "ranking", "information retrieval",
    "vector", "embedding", "indexing", "encoder", "text",
    "model", "document", "workflow", "orchestration", "backend",
    "discovery", "infrastructure", "algorithm", "representation",
    "open-source", "library",
]


def jd_overlap_score(skill: str, jd_phrases: list[str]) -> tuple[int, list[str]]:
    """Count how many JD phrases appear in the skill string (case-insensitive)."""
    s = skill.lower()
    matches = [p for p in jd_phrases if p.lower() in s]
    return len(matches), matches


def compute_suspicion_score(skill: str) -> float:
    """
    Heuristic: how 'on-the-nose' is this skill string relative to the JD?
    0.0 = generic / unrelated, 1.0 = directly paraphrases JD language.
    """
    count, _ = jd_overlap_score(skill, JD_KEY_PHRASES)
    # Each word in the skill string represents specificity
    words = skill.lower().replace("-", " ").replace("&", " ").split()
    specificity = count / len(words) if words else 0.0
    return min(1.0, specificity)


def run_leakage_test(audit_results_path: str) -> dict:
    if not Path(audit_results_path).exists():
        raise FileNotFoundError(f"Audit results not found: {audit_results_path}")

    with open(audit_results_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    fp_holder_details   = audit.get("fp_holder_details", [])
    alignment_rate      = audit.get("seniority_alignment_rate", 0.0)
    fp_skill_summary    = audit.get("fp_skill_summary", [])
    total_candidates    = audit.get("total_candidates_scanned", 100000)

    # -----------------------------------------------------------------------
    # TEST 1: Textual proximity of skill strings to JD language
    # -----------------------------------------------------------------------
    skill_jd_analysis = []
    for skill in FINGERPRINT_SKILLS:
        count, matches = jd_overlap_score(skill, JD_KEY_PHRASES)
        suspicion = compute_suspicion_score(skill)
        skill_jd_analysis.append({
            "skill":            skill,
            "jd_phrase_hits":   count,
            "matched_phrases":  matches,
            "suspicion_score":  round(suspicion, 2),
            "verdict":          "HIGH" if suspicion >= 0.5 else ("MODERATE" if suspicion >= 0.25 else "LOW"),
        })
    skill_jd_analysis.sort(key=lambda x: x["suspicion_score"], reverse=True)

    high_suspicion  = [s for s in skill_jd_analysis if s["verdict"] == "HIGH"]
    avg_suspicion   = sum(s["suspicion_score"] for s in skill_jd_analysis) / len(skill_jd_analysis)

    # -----------------------------------------------------------------------
    # TEST 2: Alignment probability under a null hypothesis
    # If we drew n=8 candidates uniformly at random from the pool, what is the
    # probability that all 8 are Senior/Staff/Lead AI/ML titled?
    # Use pool proportion: ~130/100000 = 0.0013 senior AI/ML-titled candidates.
    # -----------------------------------------------------------------------
    n_fp_holders      = len(fp_holder_details)
    senior_ai_count   = audit.get("senior_ai_ml_count", 0)
    pool_senior_rate  = 130 / total_candidates          # from master context
    # P(all n senior AI) under null = pool_senior_rate^n
    null_probability  = pool_senior_rate ** n_fp_holders if n_fp_holders > 0 else 0.0

    # -----------------------------------------------------------------------
    # TEST 3: Occurrence count distribution — unnaturally tight?
    # An organic ultra-rare tail would be roughly power-law distributed.
    # A planted set might show suspiciously uniform counts.
    # -----------------------------------------------------------------------
    occurrence_counts = [s["unique_candidates"] for s in fp_skill_summary if s["unique_candidates"] > 0]
    if occurrence_counts:
        mean_count = sum(occurrence_counts) / len(occurrence_counts)
        variance   = sum((c - mean_count) ** 2 for c in occurrence_counts) / len(occurrence_counts)
        std_dev    = math.sqrt(variance)
        cv         = std_dev / mean_count if mean_count > 0 else 0.0  # coefficient of variation
    else:
        mean_count = std_dev = cv = 0.0

    # -----------------------------------------------------------------------
    # Final assessment
    # -----------------------------------------------------------------------
    # Criteria for "likely planted":
    #   - Average suspicion score >= 0.40 (skill strings strongly echo JD language)
    #   - Alignment rate == 1.0 AND n >= 5 (perfect alignment at small sample)
    #   - null_probability < 1e-10 (alignment impossible under null)
    # Criteria for "likely organic":
    #   - Some fingerprint holders are NOT senior AI/ML (alignment < 1.0)
    #   - Skill strings have low average suspicion (<0.25)
    #   - cv of occurrence counts is large (power-law-like, not uniform)

    if alignment_rate == 1.0 and avg_suspicion >= 0.35 and null_probability < 1e-6:
        planting_verdict = "LIKELY PLANTED"
        confidence       = "HIGH"
    elif alignment_rate >= 0.875 and avg_suspicion >= 0.25:
        planting_verdict = "PLAUSIBLY PLANTED"
        confidence       = "MODERATE"
    else:
        planting_verdict = "INSUFFICIENT EVIDENCE TO CONCLUDE PLANTING"
        confidence       = "LOW"

    results = {
        "skill_jd_analysis":         skill_jd_analysis,
        "high_suspicion_skills":     [s["skill"] for s in high_suspicion],
        "avg_suspicion_score":       round(avg_suspicion, 3),
        "alignment_rate":            alignment_rate,
        "n_fp_holders":              n_fp_holders,
        "senior_ai_ml_count":        senior_ai_count,
        "pool_senior_ai_rate":       round(pool_senior_rate, 5),
        "null_probability_all_senior": null_probability,
        "occurrence_count_mean":     round(mean_count, 2),
        "occurrence_count_std":      round(std_dev, 2),
        "occurrence_count_cv":       round(cv, 3),
        "planting_verdict":          planting_verdict,
        "planting_confidence":       confidence,
    }

    print(f"\n=== LEAKAGE HYPOTHESIS TEST ===")
    print(f"\nTEST 1: JD textual proximity")
    print(f"  Average suspicion score (0=generic, 1=paraphrases JD): {avg_suspicion:.3f}")
    print(f"  High-suspicion skills (score >= 0.5):")
    for s in high_suspicion:
        print(f"    '{s['skill']}' -> score={s['suspicion_score']}, matched: {s['matched_phrases']}")

    print(f"\nTEST 2: Seniority alignment")
    print(f"  Alignment rate: {alignment_rate:.1%} ({senior_ai_count}/{n_fp_holders})")
    print(f"  Pool base rate of Senior AI/ML: {pool_senior_rate:.4%}")
    print(f"  P(all {n_fp_holders} holders are Senior AI/ML by chance): {null_probability:.2e}")

    print(f"\nTEST 3: Occurrence count distribution")
    print(f"  Counts: {sorted(occurrence_counts)}")
    print(f"  Mean: {mean_count:.2f}, Std: {std_dev:.2f}, CV: {cv:.3f}")
    print(f"  (Low CV = suspiciously uniform; High CV = power-law-like / more organic)")

    print(f"\nFINAL LEAKAGE VERDICT: {planting_verdict} (confidence: {confidence})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, help="Path to frequency_audit_results.json")
    args = parser.parse_args()
    run_leakage_test(args.audit)
