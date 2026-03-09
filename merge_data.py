import json
from pathlib import Path

BASE = Path("/Users/ozercemkelahmet/Desktop/schemalabsai")

# Load both
with open(BASE / "data" / "v1_production_500k.json") as f:
    synthetic = json.load(f)

with open(BASE / "data" / "kaggle_metadata.json") as f:
    kaggle = json.load(f)

# Tag sources
for d in synthetic:
    d["source"] = "synthetic"
for d in kaggle:
    d["source"] = "kaggle"

# Merge
merged = synthetic + kaggle
print(f"Synthetic: {len(synthetic):,}")
print(f"Kaggle: {len(kaggle)}")
print(f"Total: {len(merged):,}")

# Save
out = BASE / "data" / "v1_training_data.json"
with open(out, "w") as f:
    json.dump(merged, f)

size_mb = out.stat().st_size / 1024 / 1024
print(f"Saved: {out} ({size_mb:.1f} MB)")
