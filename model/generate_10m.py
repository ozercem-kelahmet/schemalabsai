"""
SchemaLabs.AI Base Model - FAST Synthetic Data Generator
Multiprocessing ile hızlandırılmış
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def generate_feature(n_rows, distribution):
    if distribution == "normal":
        return np.random.normal(np.random.uniform(-100, 100), np.random.uniform(1, 50), n_rows)
    elif distribution == "lognormal":
        return np.random.lognormal(np.random.uniform(0, 3), np.random.uniform(0.1, 1), n_rows)
    elif distribution == "uniform":
        return np.random.uniform(np.random.uniform(-100, 0), np.random.uniform(1, 100), n_rows)
    elif distribution == "exponential":
        return np.random.exponential(np.random.uniform(1, 20), n_rows)
    elif distribution == "poisson":
        return np.random.poisson(np.random.uniform(1, 50), n_rows).astype(float)
    elif distribution == "bimodal":
        mask = np.random.random(n_rows) < 0.5
        return np.where(mask, np.random.normal(-30, 8, n_rows), np.random.normal(30, 8, n_rows))
    elif distribution == "skewed_left":
        return np.random.beta(5, 2, n_rows) * 100
    elif distribution == "skewed_right":
        return np.random.beta(2, 5, n_rows) * 100
    else:
        return np.random.randn(n_rows)


def generate_single_dataset(args):
    dataset_id, n_rows, n_features, n_classes, output_dir, seed = args
    np.random.seed(seed + dataset_id)
    random.seed(seed + dataset_id)
    
    distributions = ["normal", "lognormal", "uniform", "exponential", "poisson", "bimodal", "skewed_left", "skewed_right"]
    
    X = np.column_stack([
        generate_feature(n_rows, random.choice(distributions))
        for _ in range(n_features)
    ])
    
    # Correlation
    n_corr = random.randint(0, n_features // 4)
    for _ in range(n_corr):
        if n_features >= 2:
            i, j = random.sample(range(n_features), 2)
            noise = np.random.randn(n_rows) * np.std(X[:, i]) * 0.2
            if random.random() < 0.5:
                X[:, j] = X[:, i] * random.uniform(0.5, 2) + noise
            else:
                X[:, j] = -X[:, i] * random.uniform(0.5, 2) + noise
    
    # Missing
    missing_rate = random.choice([0, 0, 0, 0.05, 0.1, 0.2])
    if missing_rate > 0:
        mask = np.random.random(X.shape) < missing_rate
        X[mask] = np.nan
    
    # Target
    weights = np.random.randn(n_features)
    scores = np.nanmean(X * weights, axis=1)
    scores = np.nan_to_num(scores, nan=0)
    noise = np.random.randn(n_rows) * np.nanstd(scores) * random.uniform(0.1, 0.4)
    scores = scores + noise
    thresholds = np.percentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1])
    y = np.digitize(scores, thresholds)
    
    # Imbalance
    if random.random() < 0.3:
        weights = np.random.dirichlet(np.ones(n_classes) * random.uniform(0.3, 2))
        y = np.random.choice(n_classes, size=n_rows, p=weights)
    
    # Outliers
    if random.random() < 0.3:
        n_outliers = int(n_rows * 0.02)
        for _ in range(n_outliers):
            row, col = np.random.randint(0, n_rows), np.random.randint(0, n_features)
            X[row, col] = np.nanmean(X[:, col]) + np.random.choice([-1, 1]) * np.nanstd(X[:, col]) * random.uniform(5, 10)
    
    columns = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    
    filepath = Path(output_dir) / f"dataset_{dataset_id:04d}.parquet"
    df.to_parquet(filepath, index=False, compression='snappy')
    
    return {
        "dataset_id": dataset_id,
        "n_rows": n_rows,
        "n_features": n_features,
        "n_classes": n_classes,
        "missing_rate": missing_rate,
        "filename": f"dataset_{dataset_id:04d}.parquet"
    }


def main():
    output_dir = Path("data/base_model_10m")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_datasets = 1000
    total_rows = 10_000_000
    seed = 42
    
    np.random.seed(seed)
    random.seed(seed)
    
    print("=" * 60)
    print("SchemaLabs.AI Base Model - FAST Data Generation")
    print(f"Target: {n_datasets} datasets, {total_rows:,} rows")
    print(f"CPUs: {cpu_count()}")
    print("=" * 60)
    
    # Row distribution
    row_dist = []
    for _ in range(n_datasets):
        size = random.choice(["small", "medium", "large", "xlarge"])
        if size == "small":
            rows = random.randint(20_000, 80_000)
        elif size == "medium":
            rows = random.randint(80_000, 180_000)
        elif size == "large":
            rows = random.randint(180_000, 350_000)
        else:
            rows = random.randint(350_000, 600_000)
        row_dist.append(rows)
    
    scale = total_rows / sum(row_dist)
    row_dist = [int(r * scale) for r in row_dist]
    
    feature_opts = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    class_opts = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
    
    tasks = [
        (i, row_dist[i], random.choice(feature_opts), random.choice(class_opts), str(output_dir), seed)
        for i in range(n_datasets)
    ]
    
    start = datetime.now()
    metadata = []
    
    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} workers\n")
    
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_single_dataset, tasks)):
            metadata.append(result)
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = (datetime.now() - start).total_seconds()
                done_rows = sum(m["n_rows"] for m in metadata)
                rate = done_rows / elapsed if elapsed > 0 else 0
                remaining = (total_rows - done_rows) / rate if rate > 0 else 0
                eta = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"
                pct = done_rows / total_rows * 100
                print(f"[{i+1:4d}/{n_datasets}] {done_rows:>12,} rows ({pct:5.1f}%) | {rate:,.0f} rows/s | ETA: {eta}")
    
    metadata.sort(key=lambda x: x["dataset_id"])
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    elapsed = (datetime.now() - start).total_seconds()
    total_generated = sum(m["n_rows"] for m in metadata)
    
    print("\n" + "=" * 60)
    print(f"COMPLETED!")
    print(f"Total: {total_generated:,} rows in {n_datasets} datasets")
    print(f"Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s")
    print(f"Rate: {total_generated / elapsed:,.0f} rows/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
