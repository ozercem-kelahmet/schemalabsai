"""
SchemaLabs.AI Base Model - Better Synthetic Data
Feature-target ilişkisi güçlü
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


def generate_single_dataset(args):
    dataset_id, n_rows, n_features, n_classes, output_dir, seed = args
    np.random.seed(seed + dataset_id)
    random.seed(seed + dataset_id)
    
    # Feature'lar
    X = np.random.randn(n_rows, n_features)
    
    # Güçlü feature-target ilişkisi
    # Her class için bir "center" oluştur
    centers = np.random.randn(n_classes, n_features) * 3
    
    # Her sample'ı bir class'a ata ve o center'a yakın yap
    y = np.random.randint(0, n_classes, n_rows)
    
    for i in range(n_rows):
        c = y[i]
        # Sample'ı center'a yaklaştır + noise
        X[i] = centers[c] + np.random.randn(n_features) * 0.5
    
    # Biraz karıştır (daha zor yap)
    noise_level = random.uniform(0.3, 0.7)
    X = X + np.random.randn(n_rows, n_features) * noise_level
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
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
        "filename": f"dataset_{dataset_id:04d}.parquet"
    }


def main():
    output_dir = Path("data/base_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1M test için 100 dataset
    n_datasets = 100
    seed = 42
    
    np.random.seed(seed)
    random.seed(seed)
    
    configs = []
    for i in range(n_datasets):
        n_rows = random.randint(500, 20000)
        n_features = random.randint(10, 200)
        n_classes = random.randint(2, 100)
        configs.append((i, n_rows, n_features, n_classes, str(output_dir), seed))
    
    print(f"Generating {n_datasets} datasets...")
    
    with Pool(cpu_count()) as pool:
        results = list(pool.imap(generate_single_dataset, configs))
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_rows = sum(r["n_rows"] for r in results)
    print(f"Done! {n_datasets} datasets, {total_rows:,} rows")


if __name__ == "__main__":
    main()
