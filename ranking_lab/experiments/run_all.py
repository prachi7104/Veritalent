"""
ranking_lab/experiments/run_all.py

Master orchestrator. Runs experiments A through E in sequence, collects 
all results, and writes ranking_comparison_report.md.

Usage:
    python ranking_lab/experiments/run_all.py

Will automatically use LLM-judged labels if available (>=200 entries),
otherwise falls back to synthetic formula labels (documented in report).
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ranking_lab.evaluation.ndcg_eval import load_gold_set
from ranking_lab.experiments.common import load_feature_store, load_labels
from ranking_lab.experiments import (
    exp_a_linear,
    exp_b_gbm_pointwise,
    exp_c_gbm_lambdarank,
    exp_d_lambdarank_synth_control,
    exp_e_ensemble,
)

REPORT_PATH = Path("ranking_lab/reports/ranking_comparison_report.md")


def _determine_production_recommendation(results: list[dict]) -> dict:
    """
    Picks the best model by NDCG@10. If LambdaRank beats linear by >=0.02,
    recommends LambdaRank. Otherwise recommends linear (stable, explainable).
    """
    linear = next(r for r in results if "A_linear" in r["experiment"])
    best_gbm = max(
        (r for r in results if "A_linear" not in r["experiment"] and "PENDING" not in r.get("status", "")),
        key=lambda r: r.get("ndcg@10", 0.0),
        default=None,
    )
    if best_gbm is None:
        return {"winner": linear["experiment"], "reason": "No GBM results available."}

    delta = best_gbm.get("ndcg@10", 0.0) - linear.get("ndcg@10", 0.0)
    if delta >= 0.02:
        return {
            "winner": best_gbm["experiment"],
            "delta_vs_linear": delta,
            "reason": f"GBM outperforms linear by {delta:+.4f} NDCG@10 — exceeds 0.02 significance threshold.",
        }
    else:
        return {
            "winner": linear["experiment"],
            "delta_vs_linear": delta,
            "reason": f"GBM advantage ({delta:+.4f} NDCG@10) is below 0.02 significance threshold. Linear baseline is preferred for stability and explainability.",
        }


def write_report(results: list[dict], recommendation: dict, label_source: str):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ranking Research Lab — Comparison Report",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Label source: `{label_source}`",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Experiment | Model | NDCG@10 | NDCG@50 | P@10 | P@50 | R@50 |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['experiment']} | {r['model']} "
            f"| {r.get('ndcg@10', 0.0):.4f} "
            f"| {r.get('ndcg@50', 0.0):.4f} "
            f"| {r.get('p@10', 0.0):.4f} "
            f"| {r.get('p@50', 0.0):.4f} "
            f"| {r.get('r@50', 0.0):.4f} |"
        )

    lines += ["", "---", "", "## Production Recommendation", ""]
    lines.append(f"**Winner: `{recommendation['winner']}`**")
    lines.append(f"> {recommendation['reason']}")

    # Stability section (from Exp C if available)
    exp_c = next((r for r in results if "C_gbm" in r["experiment"]), None)
    if exp_c and "stability" in exp_c:
        stab = exp_c["stability"]
        lines += [
            "",
            "---",
            "",
            "## Stability Check (LambdaRank, 3 Seeds)",
            "",
            f"- Seeds: {stab['seeds']}",
            f"- Mean NDCG@10 across seeds: {stab['mean_ndcg']:.4f}",
            f"- Mean Spearman r: {stab['mean_spearman_r']:.4f}",
            f"- Min Spearman r:  {stab['min_spearman_r']:.4f}",
            f"- Verdict: **{stab['verdict']}**",
        ]

    # Ablation section
    if exp_c and "ablation" in exp_c:
        lines += ["", "---", "", "## Feature Group Ablation (LambdaRank)", ""]
        lines.append(f"Baseline NDCG@10: {exp_c['ablation']['_baseline']['ndcg@10']:.4f}")
        lines.append("")
        lines.append("| Feature Group | NDCG@10 | Delta |")
        lines.append("|---|---|---|")
        for group, vals in exp_c["ablation"].items():
            if group == "_baseline":
                continue
            lines.append(f"| {group} | {vals['ndcg@10']:.4f} | {vals['ndcg@10_delta']:+.4f} |")

    # Adversarial section
    if exp_c and "adversarial" in exp_c:
        adv = exp_c["adversarial"]
        lines += ["", "---", "", "## Adversarial Stress Test (LambdaRank)", ""]
        lines.append(f"Verdict: **{adv['verdict']}**")
        lines.append(f"Legitimate candidate score: `{adv['legitimate_score']:.4f}`")
        lines.append("")
        lines.append("| Profile | Score | Delta vs Legit | Result |")
        lines.append("|---|---|---|---|")
        for name, vals in adv["profiles"].items():
            status = "✅ PASS" if vals["passed"] else "❌ FAIL"
            lines.append(
                f"| {name} | {vals['score']:.4f} | {vals['delta_vs_legit']:+.4f} | {status} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "- Experiment E (Cross-Encoder Ensemble) is **PENDING Lab 02 update**. Results marked accordingly.",
        "- Linear baseline (Exp A) is kept permanently as production fallback.",
        "- All GBM models trained with monotonic constraints: `trust_score=-1`, `skill_depth=+1`.",
        "- `industry_relevance`, `career_velocity`, and `fingerprint_flag` excluded from GBM per prior ablation.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


def main():
    print("Loading feature store and gold set...")
    feature_store = load_feature_store()
    gold_judgments = load_gold_set()

    # Determine label source for report header
    llm_path = Path("ranking_lab/labels/llm_labels.json")
    if llm_path.exists():
        with open(llm_path) as f:
            llm = json.load(f)
        label_source = f"llm_judged ({len(llm)} candidates)" if len(llm) >= 200 else "synthetic_formula (LLM incomplete)"
    else:
        label_source = "synthetic_formula"

    results = []

    print("\nRunning all experiments...")
    results.append(exp_a_linear.run(feature_store, gold_judgments))
    results.append(exp_b_gbm_pointwise.run(feature_store, gold_judgments))
    results.append(exp_c_gbm_lambdarank.run(feature_store, gold_judgments))
    results.append(exp_d_lambdarank_synth_control.run(feature_store, gold_judgments))
    results.append(exp_e_ensemble.run(feature_store, gold_judgments))

    recommendation = _determine_production_recommendation(results)

    print(f"\n{'='*60}")
    print(f"PRODUCTION RECOMMENDATION: {recommendation['winner']}")
    print(f"Reason: {recommendation['reason']}")
    print(f"{'='*60}")

    # Save raw results JSON
    raw_path = Path("ranking_lab/reports/raw_results.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Purge non-serializable objects before saving
    import copy
    safe_results = copy.deepcopy(results)
    raw_path.write_text(json.dumps(safe_results, indent=2, default=str), encoding="utf-8")

    write_report(results, recommendation, label_source)


if __name__ == "__main__":
    main()
