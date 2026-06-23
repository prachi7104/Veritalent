"""
redundancy_check.py — Is the fingerprint signal just title-tier in disguise?

For the 8 (actual count TBD) fingerprint-holding candidates, computes their rank
under two scoring conditions:
  A) Title-tier + deep-IR skill depth only (no fingerprint flag)
  B) Same scorer + fingerprint flag added as a bonus

If all fingerprint holders are already in the top N under (A), the fingerprint
contributes marginal new information — it is correlated with, not independent of,
signals the system already has.

Usage:
    python -m fingerprint_lab.analysis.redundancy_check \\
        --input <path/to/candidates.jsonl> \\
        --audit <path/to/frequency_audit_results.json>
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Deep-IR band (from master context, Section 2.5)
DEEP_IR_SKILLS = {
    "PyTorch", "TensorFlow", "NLP", "Machine Learning", "Deep Learning",
    "BM25", "Learning to Rank", "Qdrant", "Weaviate", "Milvus",
    "scikit-learn", "Elasticsearch", "OpenSearch", "LlamaIndex",
    "Haystack", "QLoRA", "PEFT", "LoRA", "pgvector", "Natural Language Processing",
}

FINGERPRINT_SKILLS = {
    "Search Backend", "Ranking Systems", "Text Encoders", "Vector Representations",
    "Content Matching", "Model Adaptation", "Information Retrieval Systems",
    "Search & Discovery", "Search Infrastructure", "Indexing Algorithms",
    "Workflow Orchestration", "Open-source ML libraries", "Document Processing",
}

PROFICIENCY_WEIGHTS = {"beginner": 0.5, "intermediate": 1.0, "advanced": 1.5, "expert": 2.0}

# Title seniority tiers (same as career_features.py)
TITLE_TIERS = {
    "staff":            6,
    "principal":        7,
    "lead":             5,
    "senior":           4,
    "machine learning": 3,
    "ml engineer":      3,
    "ai engineer":      3,
    "data scientist":   3,
    "nlp":              3,
    "engineer":         2,
}


def title_tier(title: str) -> int:
    t = title.lower()
    best = 1
    for tok, rank in TITLE_TIERS.items():
        if tok in t and rank > best:
            best = rank
    return best


def score_without_fp(candidate: dict) -> float:
    """Score using title_tier + deep-IR skill depth only. No fingerprint."""
    tier  = title_tier(candidate.get("profile", {}).get("current_title", ""))
    depth = 0.0
    for skill in candidate.get("skills", []):
        if skill.get("name", "") in DEEP_IR_SKILLS:
            w   = PROFICIENCY_WEIGHTS.get(skill.get("proficiency", "intermediate").lower(), 1.0)
            dur = float(skill.get("duration_months", 0) or 0)
            depth = max(depth, dur * w)
    # Normalise tier to a 0-1 range and combine
    return (tier / 11.0) * 0.4 + (min(depth, 200) / 200.0) * 0.6


def score_with_fp(candidate: dict, fp_bonus: float = 0.1) -> float:
    """Same as without_fp, plus a fixed bonus for holding ≥1 fingerprint skill."""
    base = score_without_fp(candidate)
    has_fp = any(s.get("name", "") in FINGERPRINT_SKILLS for s in candidate.get("skills", []))
    return base + (fp_bonus if has_fp else 0.0)


def load_iter(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_redundancy_check(input_path: str, audit_results_path: str | None = None) -> dict:
    # Load fingerprint holder IDs from audit (or compute inline)
    fp_holder_ids: set[str] = set()
    if audit_results_path and Path(audit_results_path).exists():
        with open(audit_results_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        fp_holder_ids = {d["candidate_id"] for d in audit.get("fp_holder_details", [])}

    # Score all candidates
    scores_without: list[tuple[str, float, str, bool]] = []   # (cid, score, title, is_fp)
    scores_with:    list[tuple[str, float, str, bool]] = []

    for candidate in tqdm(load_iter(input_path), desc="Scoring candidates"):
        cid   = candidate.get("candidate_id", "")
        title = candidate.get("profile", {}).get("current_title", "")
        has_fp = any(s.get("name", "") in FINGERPRINT_SKILLS for s in candidate.get("skills", []))

        if not fp_holder_ids:
            # Compute inline if no audit file
            if has_fp:
                fp_holder_ids.add(cid)

        s_no_fp = score_without_fp(candidate)
        s_with  = score_with_fp(candidate)
        scores_without.append((cid, s_no_fp, title, has_fp))
        scores_with.append((cid, s_with, title, has_fp))

    # Sort and rank
    scores_without.sort(key=lambda x: x[1], reverse=True)
    scores_with.sort(key=lambda x: x[1], reverse=True)

    rank_without = {x[0]: i + 1 for i, x in enumerate(scores_without)}
    rank_with    = {x[0]: i + 1 for i, x in enumerate(scores_with)}

    total = len(scores_without)

    fp_rank_report = []
    for cid, score, title, has_fp in scores_without:
        if has_fp:
            rw  = rank_without[cid]
            rwp = rank_with[cid]
            fp_rank_report.append({
                "candidate_id": cid,
                "title": title,
                "score_without_fp": round(score, 4),
                "rank_without_fp": rw,
                "rank_with_fp": rwp,
                "rank_change": rw - rwp,       # positive = improved rank
                "percentile_without_fp": round((1 - rw / total) * 100, 1),
            })

    fp_rank_report.sort(key=lambda x: x["rank_without_fp"])

    # Summary stats
    already_top_50  = sum(1 for r in fp_rank_report if r["rank_without_fp"] <= 50)
    already_top_200 = sum(1 for r in fp_rank_report if r["rank_without_fp"] <= 200)
    already_top_500 = sum(1 for r in fp_rank_report if r["rank_without_fp"] <= 500)

    results = {
        "total_candidates":     total,
        "fp_holder_count":      len(fp_rank_report),
        "fp_rank_details":      fp_rank_report,
        "already_top_50":       already_top_50,
        "already_top_200":      already_top_200,
        "already_top_500":      already_top_500,
        "max_rank_without_fp":  max((r["rank_without_fp"] for r in fp_rank_report), default=0),
        "min_rank_without_fp":  min((r["rank_without_fp"] for r in fp_rank_report), default=0),
    }

    print(f"\n=== REDUNDANCY CHECK RESULTS ===")
    print(f"Total candidates: {total:,}")
    print(f"Fingerprint holders: {len(fp_rank_report)}")
    print(f"\nRanks WITHOUT fingerprint flag:")
    for r in fp_rank_report:
        print(f"  {r['candidate_id']:15s} | title: {r['title']:35s} | "
              f"rank: {r['rank_without_fp']:6,} (top {r['percentile_without_fp']:.1f}%) | "
              f"rank with fp: {r['rank_with_fp']:6,} (change: {r['rank_change']:+d})")
    print(f"\nAlready in top 50 (without fp):  {already_top_50}/{len(fp_rank_report)}")
    print(f"Already in top 200 (without fp): {already_top_200}/{len(fp_rank_report)}")
    print(f"Already in top 500 (without fp): {already_top_500}/{len(fp_rank_report)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--audit",  default=None, help="Path to frequency_audit_results.json")
    args = parser.parse_args()
    run_redundancy_check(args.input, args.audit)
