import csv
import json
import os

CACHE_DIR = "explainability_lab/narratives_cache"
CSV_PATH = "submission/submission.csv"

# Load current CSV
rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

# Update narratives from cache
updated_count = 0
for row in rows:
    cid = row["candidate_id"]
    cache_path = os.path.join(CACHE_DIR, f"{cid}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("narrative") and data["narrative"] != row["reasoning"]:
                row["reasoning"] = data["narrative"]
                updated_count += 1

print(f"Updated {updated_count} rows with cached narratives.")

# Write updated CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
