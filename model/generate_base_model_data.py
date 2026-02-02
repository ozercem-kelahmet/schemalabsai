"""
SchemaLabs.AI Base Model - Synthetic Data Generator
1000 dataset, 150M row, feature/class agnostic
83 component için pattern çeşitliliği
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BaseModelDataGenerator:
    def __init__(self, output_dir="data/base_model", seed=42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        random.seed(seed)
        self.metadata = []
        
    def generate_feature(self, n_rows, distribution):
        if distribution == "normal":
            mean = np.random.uniform(-100, 100)
            std = np.random.uniform(1, 50)
            return np.random.normal(mean, std, n_rows)
        elif distribution == "lognormal":
            mean = np.random.uniform(0, 3)
            sigma = np.random.uniform(0.1, 1)
            return np.random.lognormal(mean, sigma, n_rows)
        elif distribution == "uniform":
            low = np.random.uniform(-100, 0)
            high = np.random.uniform(1, 100)
            return np.random.uniform(low, high, n_rows)
        elif distribution == "exponential":
            scale = np.random.uniform(1, 20)
            return np.random.exponential(scale, n_rows)
        elif distribution == "poisson":
            lam = np.random.uniform(1, 50)
            return np.random.poisson(lam, n_rows).astype(float)
        elif distribution == "bimodal":
            mean1 = np.random.uniform(-50, 0)
            mean2 = np.random.uniform(10, 60)
            std = np.random.uniform(3, 10)
            mask = np.random.random(n_rows) < 0.5
            data = np.where(mask, np.random.normal(mean1, std, n_rows), np.random.normal(mean2, std, n_rows))
            return data
        elif distribution == "skewed_left":
            return np.random.beta(5, 2, n_rows) * 100
        elif distribution == "skewed_right":
            return np.random.beta(2, 5, n_rows) * 100
        elif distribution == "multimodal":
            n_modes = np.random.randint(3, 6)
            means = np.random.uniform(-50, 50, n_modes)
            data = np.zeros(n_rows)
            for i in range(n_rows):
                mode = np.random.randint(0, n_modes)
                data[i] = np.random.normal(means[mode], 5)
            return data
        else:
            return np.random.randn(n_rows)
    
    def add_correlation(self, X, n_correlated_pairs):
        n_features = X.shape[1]
        if n_features < 2:
            return X
        for _ in range(min(n_correlated_pairs, n_features // 2)):
            i, j = random.sample(range(n_features), 2)
            corr_type = random.choice(["positive", "negative", "nonlinear"])
            noise = np.random.randn(len(X)) * np.std(X[:, i]) * 0.2
            if corr_type == "positive":
                X[:, j] = X[:, i] * np.random.uniform(0.5, 2) + noise
            elif corr_type == "negative":
                X[:, j] = -X[:, i] * np.random.uniform(0.5, 2) + noise
            elif corr_type == "nonlinear":
                X[:, j] = X[:, i] ** 2 / (np.abs(X[:, i]).max() + 1) + noise
        return X
    
    def add_missing(self, X, missing_rate, missing_type):
        if missing_rate == 0:
            return X
        mask = np.zeros_like(X, dtype=bool)
        if missing_type == "MCAR":
            mask = np.random.random(X.shape) < missing_rate
        elif missing_type == "MAR":
            n_cols_affected = max(1, int(X.shape[1] * 0.3))
            cols = random.sample(range(X.shape[1]), n_cols_affected)
            for col in cols:
                mask[:, col] = np.random.random(X.shape[0]) < missing_rate * 2
        elif missing_type == "MNAR":
            for col in range(X.shape[1]):
                threshold = np.percentile(X[:, col], 80)
                high_vals = X[:, col] > threshold
                mask[high_vals, col] = np.random.random(high_vals.sum()) < missing_rate * 3
        elif missing_type == "block":
            block_size = int(X.shape[0] * missing_rate)
            start = np.random.randint(0, X.shape[0] - block_size)
            n_cols = max(1, int(X.shape[1] * 0.2))
            cols = random.sample(range(X.shape[1]), n_cols)
            for col in cols:
                mask[start:start+block_size, col] = True
        X[mask] = np.nan
        return X
    
    def generate_target(self, X, n_classes, separation):
        n_rows = X.shape[0]
        if separation == "easy":
            feature_weights = np.random.randn(X.shape[1])
            scores = np.nanmean(X * feature_weights, axis=1)
            scores = np.nan_to_num(scores, nan=0)
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.percentile(scores, percentiles)
            y = np.digitize(scores, thresholds[1:-1])
        elif separation == "medium":
            feature_weights = np.random.randn(X.shape[1])
            scores = np.nanmean(X * feature_weights, axis=1)
            noise = np.random.randn(n_rows) * np.nanstd(scores) * 0.3
            scores = np.nan_to_num(scores + noise, nan=0)
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.percentile(scores, percentiles)
            y = np.digitize(scores, thresholds[1:-1])
        elif separation == "hard":
            primary = np.random.randn(X.shape[1])
            secondary = np.random.randn(X.shape[1])
            scores1 = np.nanmean(X * primary, axis=1)
            scores2 = np.nanmean(X * secondary, axis=1)
            scores = scores1 * scores2
            noise = np.random.randn(n_rows) * np.nanstd(scores) * 0.5
            scores = np.nan_to_num(scores + noise, nan=0)
            y = np.digitize(scores, np.percentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1]))
        else:
            y = np.random.randint(0, n_classes, n_rows)
        y = np.clip(y, 0, n_classes - 1)
        return y
    
    def apply_class_imbalance(self, y, imbalance_type):
        n_classes = len(np.unique(y))
        if imbalance_type == "balanced":
            return y
        elif imbalance_type == "mild":
            weights = np.random.dirichlet(np.ones(n_classes) * 2)
        elif imbalance_type == "severe":
            weights = np.random.dirichlet(np.ones(n_classes) * 0.5)
        elif imbalance_type == "long_tail":
            weights = 1 / (np.arange(n_classes) + 1) ** 1.5
            weights /= weights.sum()
        else:
            return y
        new_y = np.random.choice(n_classes, size=len(y), p=weights)
        return new_y
    
    def add_time_series_features(self, X, n_ts_features):
        n_rows = X.shape[0]
        ts_features = []
        for _ in range(n_ts_features):
            pattern = random.choice(["trend_up", "trend_down", "seasonal", "random_walk", "cyclic"])
            t = np.arange(n_rows)
            if pattern == "trend_up":
                ts = t / n_rows * np.random.uniform(10, 100) + np.random.randn(n_rows) * 5
            elif pattern == "trend_down":
                ts = -t / n_rows * np.random.uniform(10, 100) + np.random.randn(n_rows) * 5
            elif pattern == "seasonal":
                period = np.random.randint(10, 100)
                ts = np.sin(2 * np.pi * t / period) * np.random.uniform(10, 50) + np.random.randn(n_rows) * 3
            elif pattern == "random_walk":
                ts = np.cumsum(np.random.randn(n_rows))
            elif pattern == "cyclic":
                period1 = np.random.randint(10, 50)
                period2 = np.random.randint(50, 200)
                ts = np.sin(2 * np.pi * t / period1) * 20 + np.sin(2 * np.pi * t / period2) * 10
            ts_features.append(ts.reshape(-1, 1))
        if ts_features:
            return np.hstack([X] + ts_features)
        return X
    
    def add_outliers(self, X, outlier_rate):
        if outlier_rate == 0:
            return X
        n_outliers = int(X.shape[0] * X.shape[1] * outlier_rate)
        for _ in range(n_outliers):
            row = np.random.randint(0, X.shape[0])
            col = np.random.randint(0, X.shape[1])
            if np.random.random() < 0.5:
                X[row, col] = X[:, col].mean() + np.random.uniform(5, 10) * X[:, col].std()
            else:
                X[row, col] = X[:, col].mean() - np.random.uniform(5, 10) * X[:, col].std()
        return X
    
    def generate_dataset(self, dataset_id, n_rows, n_features, n_classes, config):
        distributions = ["normal", "lognormal", "uniform", "exponential", "poisson", 
                        "bimodal", "skewed_left", "skewed_right", "multimodal"]
        X = np.zeros((n_rows, n_features))
        for i in range(n_features):
            dist = random.choice(distributions)
            X[:, i] = self.generate_feature(n_rows, dist)
        n_correlated = config.get("n_correlated_pairs", random.randint(0, n_features // 3))
        X = self.add_correlation(X, n_correlated)
        n_ts = config.get("n_ts_features", random.randint(0, 3))
        if n_ts > 0:
            X = self.add_time_series_features(X, n_ts)
            n_features = X.shape[1]
        outlier_rate = config.get("outlier_rate", random.choice([0, 0, 0, 0.01, 0.02, 0.05]))
        X = self.add_outliers(X, outlier_rate)
        separation = config.get("separation", random.choice(["easy", "easy", "medium", "medium", "hard"]))
        y = self.generate_target(X, n_classes, separation)
        imbalance = config.get("imbalance", random.choice(["balanced", "balanced", "mild", "severe", "long_tail"]))
        y = self.apply_class_imbalance(y, imbalance)
        missing_rate = config.get("missing_rate", random.choice([0, 0, 0.05, 0.1, 0.2, 0.3]))
        missing_type = config.get("missing_type", random.choice(["MCAR", "MAR", "MNAR", "block"]))
        if missing_rate > 0:
            X = self.add_missing(X, missing_rate, missing_type)
        columns = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)
        df["target"] = y
        return df, {
            "dataset_id": dataset_id,
            "n_rows": n_rows,
            "n_features": n_features,
            "n_classes": n_classes,
            "separation": separation,
            "imbalance": imbalance,
            "missing_rate": missing_rate,
            "missing_type": missing_type if missing_rate > 0 else None,
            "outlier_rate": outlier_rate,
            "n_correlated_pairs": n_correlated,
            "n_ts_features": n_ts
        }
    
    def generate_all(self, n_datasets=1000, total_rows=150_000_000):
        print("=" * 60)
        print("SchemaLabs.AI Base Model Data Generation")
        print("=" * 60)
        print(f"Target: {n_datasets} datasets, {total_rows:,} total rows")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
        
        avg_rows = total_rows // n_datasets
        row_distribution = []
        for i in range(n_datasets):
            size_category = random.choice(["small", "medium", "large", "xlarge"])
            if size_category == "small":
                rows = random.randint(10_000, 50_000)
            elif size_category == "medium":
                rows = random.randint(50_000, 150_000)
            elif size_category == "large":
                rows = random.randint(150_000, 300_000)
            else:
                rows = random.randint(300_000, 500_000)
            row_distribution.append(rows)
        
        current_total = sum(row_distribution)
        scale_factor = total_rows / current_total
        row_distribution = [int(r * scale_factor) for r in row_distribution]
        
        feature_options = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
        class_options = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
        
        start_time = datetime.now()
        total_generated = 0
        
        for i in range(n_datasets):
            n_rows = row_distribution[i]
            n_features = random.choice(feature_options)
            n_classes = random.choice(class_options)
            
            config = {}
            df, meta = self.generate_dataset(i, n_rows, n_features, n_classes, config)
            
            filename = f"dataset_{i:04d}.parquet"
            filepath = self.output_dir / filename
            df.to_parquet(filepath, index=False)
            
            meta["filename"] = filename
            meta["actual_rows"] = len(df)
            self.metadata.append(meta)
            
            total_generated += len(df)
            
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_generated / elapsed if elapsed > 0 else 0
                eta = (total_rows - total_generated) / rate if rate > 0 else 0
                eta_str = f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
                pct = total_generated / total_rows * 100
                print(f"[{i+1:4d}/{n_datasets}] {total_generated:>12,} rows ({pct:5.1f}%) | Rate: {rate:,.0f} rows/s | ETA: {eta_str}")
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print("=" * 60)
        print(f"COMPLETED!")
        print(f"Total: {total_generated:,} rows in {n_datasets} datasets")
        print(f"Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s")
        print(f"Metadata: {metadata_path}")
        print("=" * 60)
        
        return self.metadata


def main():
    generator = BaseModelDataGenerator(output_dir="data/base_model")
    metadata = generator.generate_all(n_datasets=1000, total_rows=150_000_000)
    
    features = [m["n_features"] for m in metadata]
    classes = [m["n_classes"] for m in metadata]
    rows = [m["actual_rows"] for m in metadata]
    
    print("\nSUMMARY:")
    print(f"  Features: min={min(features)}, max={max(features)}, avg={np.mean(features):.0f}")
    print(f"  Classes:  min={min(classes)}, max={max(classes)}, avg={np.mean(classes):.0f}")
    print(f"  Rows:     min={min(rows):,}, max={max(rows):,}, avg={np.mean(rows):,.0f}")
    print(f"  Total:    {sum(rows):,} rows")


if __name__ == "__main__":
    main()
