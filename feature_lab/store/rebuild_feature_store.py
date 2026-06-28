"""
rebuild_feature_store.py — computes and persists all features for the candidates.
Handles both JSON array (.json) and JSON Lines (.jsonl) inputs.
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import feature_lab  # noqa: F401
from feature_lab.features.feature_registry import registry


def load_candidates(filepath: str):
    path = Path(filepath)
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data


def run_rebuild_store(input_path: str, output_path: str):
    registered = [f.name for f in registry.get_all_features()]
    print(f"Registry loaded with {len(registered)} features: {registered}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    candidates = list(load_candidates(input_path))
    print(f"Loaded {len(candidates)} candidates from {input_path}")

    with open(out_file, "w", encoding="utf-8") as f:
        for candidate in tqdm(candidates):
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
    parser = argparse.ArgumentParser(description="Rebuild feature store")
    parser.add_argument("--input", required=True, help="Path to candidates file")
    parser.add_argument("--output", required=True, help="Path to output feature_store.jsonl")
    args = parser.parse_args()

    run_rebuild_store(args.input, args.output)
