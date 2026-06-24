"""
ranking_lab/experiments/common.py

Shared data loading utilities for all experiments.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ranking_lab.models.monotonic_constraints import TRAINING_FEATURES

FEATURE_STORE_PATH = Path("feature_lab/store/feature_store.jsonl")
LLM_LABELS_PATH = Path("ranking_lab/labels/llm_labels.json")
SYNTH_LABELS_PATH = Path("ranking_lab/labels/synthetic_formula_labels.json")
STRATIFIED_SAMPLE_PATH = Path("ranking_lab/labels/stratified_sample.json")


def load_feature_store() -> dict:
    """Load full feature store as {candidate_id: feature_dict}."""
    store = {}
    with open(FEATURE_STORE_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            store[rec["candidate_id"]] = rec
    return store


def load_labels() -> dict:
    """
    Load labels. Prefers LLM-judged labels if available and complete
    (at least 200 entries). Falls back to synthetic formula labels.
    Falls back gracefully and prints which source is being used.
    """
    if LLM_LABELS_PATH.exists():
        with open(LLM_LABELS_PATH, encoding="utf-8") as f:
            llm = json.load(f)
        if len(llm) >= 200:
            print(f"[Labels] Using LLM-judged labels ({len(llm)} candidates).")
            return llm
        else:
            print(f"[Labels] LLM labels incomplete ({len(llm)} entries). Using synthetic fallback.")
    else:
        print("[Labels] LLM label file not found. Using synthetic formula labels (fallback).")

    with open(SYNTH_LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_training_matrix(feature_store: dict, labels: dict):
    """
    Builds (X, y, candidate_ids) for the training set (labeled candidates).
    Only includes candidates that:
      - appear in the labels dict
      - have a feature store entry
    """
    X, y, ids = [], [], []
    for cid, label in labels.items():
        if cid not in feature_store:
            continue
        feats = feature_store[cid]
        row = [float(feats.get(f, 0.0) or 0.0) for f in TRAINING_FEATURES]
        X.append(row)
        y.append(int(label))
        ids.append(cid)
    return np.array(X), np.array(y), ids


def build_eval_matrix(feature_store: dict, candidate_ids: list[str]):
    """
    Builds (X, candidate_ids) for evaluation over a set of candidates.
    Candidates missing from the feature store are skipped.
    """
    X, ids = [], []
    for cid in candidate_ids:
        if cid not in feature_store:
            continue
        feats = feature_store[cid]
        row = [float(feats.get(f, 0.0) or 0.0) for f in TRAINING_FEATURES]
        X.append(row)
        ids.append(cid)
    return np.array(X), ids
