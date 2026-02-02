"""
SchemaLabs.AI Base Model Data - %99 hedef
"""
import numpy as np
import pandas as pd
from pathlib import Path
import random
import json
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def generate_single_dataset(dataset_id, n_rows, n_features, n_classes, output_dir, seed):
    np.random.seed(seed + dataset_id)
    random.seed(seed + dataset_id)
    
    # n_informative en az n_classes kadar olmalı
    n_informative = min(n_features - 1, max(n_classes + 2, n_features // 2))
    n_redundant = min(5, n_features - n_informative - 1)
    
    try:
        X, y = make_classification(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=max(0, n_redundant),
            n_clusters_per_class=1,
            n_classes=n_classes,
            class_sep=2.5,
            flip_y=0.01,
            random_state=seed + dataset_id
        )
    except Exception as e:
        print(f"Dataset {dataset_id} failed: {e}, using fallback")
        # Fallback - basit cluster-based
        centers = np.random.randn(n_classes, n_features) * 2
        y = np.random.randint(0, n_classes, n_rows)
        X = np.zeros((n_rows, n_features))
        for i in range(n_rows):
            X[i] = centers[y[i]] + np.random.randn(n_features) * 0.5
    
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
    
    results = []
    for i in range(n_datasets):
        n_rows = random.randint(500, 20000)
        n_features = random.randint(10, 200)
        # n_classes en fazla n_features/2 olsun
        max_classes = min(30, n_features // 2)
        n_classes = random.randint(2, max(2, max_classes))
        
        result = generate_single_dataset(i, n_rows, n_features, n_classes, str(output_dir), seed)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i+1}/{n_datasets}")
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_rows = sum(r["n_rows"] for r in results)
    print(f"Done! {n_datasets} datasets, {total_rows:,} rows")
    
    # Test birkaç dataset
    print("\nTesting RF accuracy:")
    for i in [0, 1, 2, 3, 4]:
        df = pd.read_parquet(f'{output_dir}/dataset_{i:04d}.parquet')
        X = df.drop(columns=['target']).values
        y = df.target.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        print(f'  Dataset {i}: {df.target.nunique()} cls, RF={rf.score(X_test, y_test)*100:.1f}%')


if __name__ == "__main__":
    main()
