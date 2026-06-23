# Retrieval Research Lab

This is a self-contained, reproducible experimentation framework for the candidate-retrieval stage of the Redrob Hackathon ranking engine.

## Objective
Determine the best-performing retrieval configuration for surfacing relevant candidates from the full pool given a structured Job Description (JD) query. 

## Structure
- `data/`: Candidate loading, schema enforcement, and gold set generation.
- `indexing/`: Index builders for BM25, Dense embeddings, Multi-vector embeddings, and Skill Co-occurrence graph.
- `fusion/`: Result combiners (Reciprocal Rank Fusion, Learned Fusion).
- `reranking/`: Cross-encoder reranking over fused results.
- `experiments/`: Experiment implementations and the main orchestrator (`run_all.py`).
- `evaluation/`: Metrics computation.
- `tests/`: Unit tests for constraints and logic.
- `reports/`: Markdown output comparing retrieval configurations.

## Setup and Running

1. Install dependencies:
```bash
pip install -r requirements.txt # (sentence-transformers, rank_bm25, networkx, lightgbm, scikit-learn, pandas, pytest, tabulate)
```

2. Generate the Gold Set (if not already present):
```bash
python -m retrieval_lab.data.sample_validation_set dataset/candidates.jsonl retrieval_lab/evaluation/gold_set.json
```

3. Run experiments and generate report:
```bash
$env:PYTHONPATH="." 
python retrieval_lab/experiments/run_all.py
```

4. View `retrieval_lab/reports/retrieval_comparison_report.md` for findings.

## Constraints strictly enforced
- `github_activity_score` and `offer_acceptance_rate` of `-1` are explicitly treated as `NaN`.
- `education.start_year` and `education.end_year` are never queried due to high unreliability.

## Limitations
- The gold set is constructed using an inspectable heuristic (matching AI/ML Senior titles and deep-IR skill sets) for baseline comparative purposes, not human-verified ground truth.
- Local execution uses a large slice instead of the full 100K candidates to keep build times reasonable unless running on a GPU.
