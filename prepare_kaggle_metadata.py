#!/usr/bin/env python3
"""
Kaggle raw 96 dataset -> V1 metadata format
"""
import json, os, random
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(os.path.expanduser("~/Desktop/schemalabsai"))
RAW_DIR = BASE / "data" / "raw"
OUTPUT = BASE / "data" / "kaggle_metadata.json"
SECTOR_LIST = BASE / "data" / "sector_list_10000.json"

# Load sector list for SBERT matching
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer("all-MiniLM-L6-v2")

with open(SECTOR_LIST) as f:
    sector_data = json.load(f)
    ALL_SECTORS = sector_data["sectors"]
    HIERARCHY = sector_data["hierarchy"]

MAIN_SECTORS = sorted(HIERARCHY.keys())

# Pre-compute sector embeddings
print("Computing sector embeddings...")
main_embs = sbert.encode(MAIN_SECTORS, convert_to_numpy=True)

def detect_sector(column_names):
    """SBERT confidence-weighted voting"""
    sector_embs = np.load(BASE / "data" / "sector_embeddings_10000.npy")
    
    # Individual column embeddings
    votes = {}
    for col in column_names[:20]:
        col_clean = col.lower().replace("_", " ")
        col_emb = sbert.encode([col_clean], convert_to_numpy=True)[0]
        
        # Cosine similarity with main sectors
        sims = np.dot(main_embs, col_emb) / (
            np.linalg.norm(main_embs, axis=1) * np.linalg.norm(col_emb) + 1e-8
        )
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        margin = best_sim - np.sort(sims)[-2]
        weight = 1.0 + margin * 15
        
        sector = MAIN_SECTORS[best_idx]
        votes[sector] = votes.get(sector, 0) + weight
    
    # Context sentence
    context = " ".join(col.lower().replace("_", " ") for col in column_names[:15])
    ctx_emb = sbert.encode([context], convert_to_numpy=True)[0]
    sims = np.dot(main_embs, ctx_emb) / (
        np.linalg.norm(main_embs, axis=1) * np.linalg.norm(ctx_emb) + 1e-8
    )
    best_idx = np.argmax(sims)
    sector = MAIN_SECTORS[best_idx]
    votes[sector] = votes.get(sector, 0) + 2.0
    
    # Best sector
    if votes:
        return max(votes, key=votes.get)
    return "manufacturing"

# Process all Kaggle folders
print(f"Processing {RAW_DIR}...")
datasets = []
errors = 0
skipped = 0

for folder in sorted(RAW_DIR.iterdir()):
    if not folder.is_dir():
        continue
    
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        skipped += 1
        continue
    
    csv_file = csvs[0]  # Take first CSV
    try:
        df = pd.read_csv(csv_file, nrows=200, on_bad_lines="skip", encoding="utf-8")
    except:
        try:
            df = pd.read_csv(csv_file, nrows=200, on_bad_lines="skip", encoding="latin-1")
        except:
            errors += 1
            continue
    
    if len(df) < 10 or len(df.columns) < 3:
        skipped += 1
        continue
    
    columns = list(df.columns)
    
    # Detect sector
    sector = detect_sector(columns)
    
    # Sample rows (10 rows)
    sample_df = df.head(10)
    sample_rows = []
    for _, row in sample_df.iterrows():
        sample_row = []
        for val in row:
            if pd.isna(val):
                sample_row.append("")
            else:
                sample_row.append(str(val))
        sample_rows.append(sample_row)
    
    # Count missing
    missing_ratio = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
    
    # Count classes (if last column looks like target)
    n_classes = 0
    for col in reversed(columns):
        if df[col].nunique() <= 50 and df[col].nunique() >= 2:
            n_classes = int(df[col].nunique())
            break
    if n_classes == 0:
        n_classes = 2
    
    datasets.append({
        "columns": columns,
        "sample_rows": sample_rows,
        "sector": sector,
        "main_sector": sector,
        "n_rows": int(len(df)),
        "n_cols": len(columns),
        "n_classes": n_classes,
        "missing_ratio": round(missing_ratio, 3),
        "balance": "unknown",
        "folder": folder.name,
        "source": "kaggle",
    })
    
    print(f"  OK {folder.name[:40]:40s} -> {sector:20s} cols={len(columns)} rows={len(df)}")

print(f"\nProcessed: {len(datasets)}, Errors: {errors}, Skipped: {skipped}")

# Sector distribution
from collections import Counter
sector_counts = Counter(d["sector"] for d in datasets)
print(f"\nSector distribution:")
for s, c in sorted(sector_counts.items(), key=lambda x: -x[1]):
    print(f"  {s:25s}: {c}")

# Save
with open(OUTPUT, "w") as f:
    json.dump(datasets, f, indent=2)

size_kb = OUTPUT.stat().st_size / 1024
print(f"\nSaved: {OUTPUT} ({size_kb:.0f} KB)")
print(f"Total Kaggle datasets: {len(datasets)}")
