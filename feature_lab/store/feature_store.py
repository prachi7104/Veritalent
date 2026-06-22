"""
Feature Store — computes and persists all features for the full candidate pool.
Versioned by feature set hash.

Usage:
    python -m feature_lab.store.feature_store \\
        --input  <path/to/candidates.jsonl> \\
        --output <path/to/feature_store.jsonl>
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# This import triggers ALL feature-module imports via feature_lab/__init__.py,
# which populates the global registry singleton before we call compute_all().
import feature_lab  # noqa: F401 — side-effect import
from feature_lab.features.feature_registry import registry


def load_candidates_iter(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run_feature_store(input_path: str, output_path: str):
    registered = [f.name for f in registry.get_all_features()]
    print(f"Registry loaded with {len(registered)} features: {registered}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for candidate in tqdm(load_candidates_iter(input_path)):
            try:
                candidate_id = candidate.get("candidate_id")
                if not candidate_id:
                    error_count += 1
                    continue

                features = registry.compute_all(candidate)
                features["candidate_id"] = candidate_id
                f.write(json.dumps(features) + "\n")
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error processing {candidate.get('candidate_id', 'unknown')}: {e}")

    print(f"Done. Success: {success_count}, Errors: {error_count}")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate feature store")
    parser.add_argument("--input", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output", required=True, help="Path to output feature_store.jsonl")
    args = parser.parse_args()

    run_feature_store(args.input, args.output)
