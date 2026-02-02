"""
SchemaLabs.AI Base Model - Balanced Realistic Data
RF ~70-85% alacak zorlukta
"""
import numpy as np
import pandas as pd
from pathlib import Path
import random
import json
from sklearn.datasets import make_classification
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def generate_single_dataset(args):
    dataset_id, n_rows, n_features, n_classes, output_dir, seed = args
    np.random.seed(seed + dataset_id)
    random.seed(seed + dataset_id)
    
    n_informative = max(2, n_features // 2)
    n_redundant = min(n_features // 4, n_features - n_informative - 1)
    
    try:
        X, y = make_classification(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=1,
            n_classes=n_classes,
            class_sep=random.uniform(1.5, 2.5),  # Daha ayrık class'lar
            flip_y=random.uniform(0.0, 0.02),    # Az label noise
            random_state=seed + dataset_id
        )
    except:
        X = np.random.randn(n_rows, n_features)
        y = np.random.randint(0, n_classes, n_rows)
    
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
    
    n_datasets = 100
    seed = 42
    
    np.random.seed(seed)
    random.seed(seed)
    
    configs = []
    for i in range(n_datasets):
        n_rows = random.randint(500, 20000)
        n_features = random.randint(10, 200)
        n_classes = random.randint(2, 30)  # Max 30 class
        configs.append((i, n_rows, n_features, n_classes, str(output_dir), seed))
    
    print(f"Generating {n_datasets} balanced datasets...")
    
    with Pool(cpu_count()) as pool:
        results = list(pool.imap(generate_single_dataset, configs))
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_rows = sum(r["n_rows"] for r in results)
    print(f"Done! {n_datasets} datasets, {total_rows:,} rows")


if __name__ == "__main__":
    main()
