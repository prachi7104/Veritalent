import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval_lab.indexing.dense_index import DenseIndex
from retrieval_lab.data.loaders import load_candidates

# Load the full dataset 
print("Loading candidates...")
dataset_path = "dataset/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
candidates = load_candidates(dataset_path, apply_funnel=False)
print(f"Loaded {len(candidates)} candidates.")

# Build the dense index
dense_index = DenseIndex()
dense_index.build(candidates, batch_size=256)

# Ensure cache dir exists
cache_dir = "retrieval_lab/cache"
os.makedirs(cache_dir, exist_ok=True)

# Save
dense_index.save(f"{cache_dir}/dense.npz")
print("Index successfully rebuilt.")
