"""
boundary_sensitivity.py — Does the fingerprint pattern hold under different cutoffs?

Tests whether the fingerprint candidate set and skill set remain stable when the
"ultra-rare" threshold is varied: ≤3, ≤5, ≤7 (default), ≤10, ≤15, ≤20 occurrences.

A robust signal should produce a stable core set of candidates across thresholds.
Fragility to small threshold changes weakens the case for organic, robust signal.

Usage:
    python -m fingerprint_lab.analysis.boundary_sensitivity \\
        --input <path/to/candidates.jsonl>
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

FINGERPRINT_SKILLS = {
    "Search Backend", "Ranking Systems", "Text Encoders", "Vector Representations",
    "Content Matching", "Model Adaptation", "Information Retrieval Systems",
    "Search & Discovery", "Search Infrastructure", "Indexing Algorithms",
    "Workflow Orchestration", "Open-source ML libraries", "Document Processing",
}

SENIOR_TITLE_MARKERS = {"senior", "staff", "lead", "principal", "head", "chief"}
AI_ML_TITLE_MARKERS  = {
    "ai", "ml", "machine learning", "artificial intelligence",
    "data scientist", "nlp", "applied scientist",
}

THRESHOLDS = [3, 5, 7, 10, 15, 20]


def is_senior_ai_ml(title: str) -> bool:
    t = title.lower()
    return any(m in t for m in SENIOR_TITLE_MARKERS) and any(m in t for m in AI_ML_TITLE_MARKERS)


def load_iter(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_boundary_sensitivity(input_path: str) -> dict:
    # First pass: build skill -> {cid -> title} mapping
    skill_to_candidates: dict[str, dict[str, str]] = defaultdict(dict)

    for candidate in tqdm(load_iter(input_path), desc="Scanning pool"):
        cid   = candidate.get("candidate_id", "")
        title = candidate.get("profile", {}).get("current_title", "")
        for skill in candidate.get("skills", []):
            name = skill.get("name", "").strip()
            if name:
                skill_to_candidates[name][cid] = title

    skill_counts = {s: len(cands) for s, cands in skill_to_candidates.items()}

    results_by_threshold = {}
    for threshold in THRESHOLDS:
        # Skills under this threshold
        rare_skills     = {s for s, c in skill_counts.items() if 1 <= c <= threshold}
        fp_under        = rare_skills & FINGERPRINT_SKILLS

        # Candidates holding ≥1 rare skill
        rare_holders: dict[str, str] = {}
        for skill in fp_under:
            for cid, title in skill_to_candidates[skill].items():
                rare_holders[cid] = title

        # Seniority alignment
        senior_count = sum(1 for t in rare_holders.values() if is_senior_ai_ml(t))
        alignment    = senior_count / len(rare_holders) if rare_holders else 0.0

        results_by_threshold[threshold] = {
            "threshold": threshold,
            "total_rare_skills":       len(rare_skills),
            "fp_skills_under_cutoff":  sorted(fp_under),
            "fp_skills_count":         len(fp_under),
            "fp_holder_count":         len(rare_holders),
            "fp_holder_ids":           sorted(rare_holders.keys()),
            "fp_holder_titles":        list(rare_holders.values()),
            "senior_ai_ml_count":      senior_count,
            "seniority_alignment":     round(alignment, 3),
        }

    # Print comparison table
    print(f"\n=== BOUNDARY SENSITIVITY RESULTS ===")
    print(f"{'Cutoff':>8} | {'FP skills':>10} | {'Holders':>8} | {'Senior AI/ML':>13} | {'Alignment':>10} | Holder IDs")
    print("-" * 100)
    for t, r in results_by_threshold.items():
        ids_str = ", ".join(r["fp_holder_ids"][:5])
        if len(r["fp_holder_ids"]) > 5:
            ids_str += f" ... (+{len(r['fp_holder_ids'])-5} more)"
        print(f"{t:>8} | {r['fp_skills_count']:>10} | {r['fp_holder_count']:>8} | "
              f"{r['senior_ai_ml_count']:>13} | {r['seniority_alignment']:>10.1%} | {ids_str}")

    # Stability analysis: is the core holder set consistent?
    baseline = set(results_by_threshold[7]["fp_holder_ids"])
    print(f"\nStability (core set at threshold 7):")
    for t, r in results_by_threshold.items():
        s = set(r["fp_holder_ids"])
        overlap = len(s & baseline)
        print(f"  Threshold {t:2d}: {overlap}/{len(baseline)} core candidates retained, "
              f"{len(s - baseline)} new candidates added")

    return results_by_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    run_boundary_sensitivity(args.input)
