"""
frequency_audit.py — Independent skill frequency re-derivation.

Re-counts every skill's occurrence across the FULL 100k candidate pool from scratch.
Does NOT reuse any prior cached count. This is an adversarial sanity check: if the
original finding (13 skills, 1-7 occurrences, held by 8 candidates, all senior-titled)
was a sampling artifact or computation error, this will catch it.

Fingerprint skill set (from master context, Section 2.5):
  "Search Backend", "Ranking Systems", "Text Encoders", "Vector Representations",
  "Content Matching", "Model Adaptation", "Information Retrieval Systems",
  "Search & Discovery", "Search Infrastructure", "Indexing Algorithms",
  "Workflow Orchestration", "Open-source ML libraries", "Document Processing"

Usage:
    python -m fingerprint_lab.analysis.frequency_audit \\
        --input <path/to/candidates.jsonl> \\
        --output <path/to/frequency_audit_results.json>
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

CLAIMED_FINGERPRINT_SKILLS = {
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


def is_senior_ai_ml_title(title: str) -> bool:
    t = title.lower()
    has_seniority = any(m in t for m in SENIOR_TITLE_MARKERS)
    has_ai_ml     = any(m in t for m in AI_ML_TITLE_MARKERS)
    return has_seniority and has_ai_ml


def load_iter(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_frequency_audit(input_path: str, output_path: str | None = None) -> dict:
    """
    Streams the full candidate pool and computes:
      1. skill_counts: total occurrences of every skill string across all candidates
      2. skill_candidate_counts: number of unique candidates who list each skill
      3. fingerprint_holders: candidate_ids + titles of candidates holding ≥1 fingerprint skill
      4. per-skill fingerprint detail: exact count, which candidates, their titles
    """
    skill_counts: dict[str, int]             = defaultdict(int)       # total skill mentions
    skill_candidate_counts: dict[str, set]   = defaultdict(set)       # unique candidates per skill
    fingerprint_detail: dict[str, dict]      = {}                     # detail for fp skills

    total_candidates = 0
    for candidate in tqdm(load_iter(input_path), desc="Auditing skills"):
        total_candidates += 1
        cid   = candidate.get("candidate_id", "")
        title = candidate.get("profile", {}).get("current_title", "")

        for skill in candidate.get("skills", []):
            name = skill.get("name", "").strip()
            if not name:
                continue
            skill_counts[name] += 1
            skill_candidate_counts[name].add(cid)

            if name in CLAIMED_FINGERPRINT_SKILLS:
                if name not in fingerprint_detail:
                    fingerprint_detail[name] = {"candidates": {}}
                fingerprint_detail[name]["candidates"][cid] = title

    # Compute frequency bands for context
    freq_distribution = {
        "ultra_rare_1_7":    [],
        "rare_8_50":         [],
        "moderate_51_1000":  [],
        "common_1001_plus":  [],
    }
    for skill, cnt in skill_counts.items():
        if cnt <= 7:
            freq_distribution["ultra_rare_1_7"].append((skill, cnt))
        elif cnt <= 50:
            freq_distribution["rare_8_50"].append((skill, cnt))
        elif cnt <= 1000:
            freq_distribution["moderate_51_1000"].append((skill, cnt))
        else:
            freq_distribution["common_1001_plus"].append((skill, cnt))

    # Candidates who hold ≥1 fingerprint skill
    all_fp_holders: dict[str, str] = {}
    for detail in fingerprint_detail.values():
        for cid, title in detail["candidates"].items():
            all_fp_holders[cid] = title

    # Check whether each fp skill is actually in the ultra-rare bucket
    fp_discrepancies = []
    for skill in CLAIMED_FINGERPRINT_SKILLS:
        actual_count = len(skill_candidate_counts.get(skill, set()))
        if actual_count == 0:
            fp_discrepancies.append(f"  SKILL ABSENT: '{skill}' has 0 occurrences (not present in pool)")
        elif actual_count > 7:
            fp_discrepancies.append(
                f"  COUNT MISMATCH: '{skill}' has {actual_count} unique-candidate occurrences "
                f"(claimed: 1-7)"
            )

    # Seniority alignment check
    fp_holder_details = []
    for cid, title in all_fp_holders.items():
        senior_ai = is_senior_ai_ml_title(title)
        fp_holder_details.append({
            "candidate_id": cid,
            "title": title,
            "is_senior_ai_ml": senior_ai,
        })

    senior_count = sum(1 for d in fp_holder_details if d["is_senior_ai_ml"])
    alignment_rate = senior_count / len(fp_holder_details) if fp_holder_details else 0.0

    # Per-fingerprint-skill summary
    fp_skill_summary = []
    for skill in sorted(CLAIMED_FINGERPRINT_SKILLS):
        cands = fingerprint_detail.get(skill, {}).get("candidates", {})
        fp_skill_summary.append({
            "skill": skill,
            "unique_candidates": len(cands),
            "candidate_titles": list(cands.values()),
        })
    fp_skill_summary.sort(key=lambda x: x["unique_candidates"])

    results = {
        "total_candidates_scanned": total_candidates,
        "total_unique_skills": len(skill_counts),
        "claimed_fingerprint_skills": sorted(CLAIMED_FINGERPRINT_SKILLS),
        "fingerprint_skills_found": sorted(fingerprint_detail.keys()),
        "fingerprint_skills_absent": sorted(CLAIMED_FINGERPRINT_SKILLS - set(fingerprint_detail.keys())),
        "fp_skill_summary": fp_skill_summary,
        "total_fp_holders": len(all_fp_holders),
        "fp_holder_details": fp_holder_details,
        "seniority_alignment_rate": alignment_rate,
        "senior_ai_ml_count": senior_count,
        "discrepancies_vs_claimed": fp_discrepancies,
        "ultra_rare_skill_count": len(freq_distribution["ultra_rare_1_7"]),
        "ultra_rare_skills": sorted(freq_distribution["ultra_rare_1_7"], key=lambda x: x[1]),
        "freq_band_summary": {
            k: len(v) for k, v in freq_distribution.items()
        },
    }

    print(f"\n=== FREQUENCY AUDIT RESULTS ===")
    print(f"Total candidates scanned: {total_candidates:,}")
    print(f"Total unique skill strings: {results['total_unique_skills']}")
    print(f"Ultra-rare skills (1-7 occurrences): {results['ultra_rare_skill_count']}")
    print(f"\nFingerprint skills found: {len(results['fingerprint_skills_found'])}/13")
    if results["fingerprint_skills_absent"]:
        print(f"ABSENT: {results['fingerprint_skills_absent']}")
    print(f"\nFingerprint holders: {results['total_fp_holders']}")
    print(f"Senior AI/ML alignment: {senior_count}/{len(fp_holder_details)} = {alignment_rate:.1%}")
    if fp_discrepancies:
        print(f"\n*** DISCREPANCIES vs. claimed numbers ***")
        for d in fp_discrepancies:
            print(d)
    else:
        print("\nNo discrepancies — all claimed fingerprint skills match documented ranges.")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            # Convert sets to lists for JSON serialisation
            json.dump(results, f, indent=2, default=list)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent skill frequency audit")
    parser.add_argument("--input",  required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output", default=None,  help="Optional output JSON path")
    args = parser.parse_args()
    run_frequency_audit(args.input, args.output)
