import os
from analytics_engine import detect_analytics_type, generate_analytics
os.environ["FLASK_SKIP_DOTENV"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify
from datetime import datetime
import json
import math

class NaNSafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
        return super().default(obj)
    
    def encode(self, obj):
        def clean_nan(o):
            if isinstance(o, dict):
                return {k: clean_nan(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [clean_nan(i) for i in o]
            elif isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return 0.0
            return o
        return super().encode(clean_nan(obj))

app_json_encoder = NaNSafeEncoder
import torch
torch.backends.mkldnn.enabled = True
torch.set_float32_matmul_precision("medium")
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from model import TabularFoundationModel, TabularFoundationModelMIRAS
import os
import sys
import time
import threading
import tempfile
import glob
from pathlib import Path
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans



def detect_and_fix_type_mismatch(df):
    report = {"fixed_columns": [], "total_fixes": 0}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_count = 0
            non_numeric_values = []
            
            for val in df[col].dropna().head(100):
                try:
                    float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except:
                    non_numeric_values.append(val)
            
            total = len(df[col].dropna().head(100))
            if total > 0 and numeric_count / total > 0.8:
                def safe_convert(x):
                    try:
                        return float(str(x).replace(',', '').replace('$', '').replace('%', ''))
                    except:
                        return np.nan
                
                df[col] = df[col].apply(safe_convert)
                report["fixed_columns"].append({"column": col, "non_numeric_values": non_numeric_values[:5], "converted_to": "numeric"})
                report["total_fixes"] += len(non_numeric_values)
    
    return df, report


def detect_scale_inconsistency(df):
    report = {"scaled_columns": [], "total_adjustments": 0}
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
        
        upper_bound = Q3 + 3 * IQR
        extreme_high = series[series > upper_bound * 10]
        
        if len(extreme_high) > 0:
            median_val = series.median()
            for idx in extreme_high.index:
                val = df.loc[idx, col]
                if val > median_val * 500:
                    ratio = val / median_val if median_val != 0 else 0
                    if 500 < ratio < 1500:
                        df.loc[idx, col] = val / 1000
                        report["total_adjustments"] += 1
                    elif 500000 < ratio < 1500000:
                        df.loc[idx, col] = val / 1000000
                        report["total_adjustments"] += 1
            
            if report["total_adjustments"] > 0:
                report["scaled_columns"].append({"column": col, "issue": "scale_mismatch", "median": float(median_val)})
    
    return df, report


def smart_data_cleaning(df, use_midas=False, midas_model=None):
    cleaning_report = {"original_shape": df.shape, "type_fixes": {}, "scale_fixes": {}, "missing_fixes": {}, "final_shape": None}
    
    df, type_report = detect_and_fix_type_mismatch(df)
    cleaning_report["type_fixes"] = type_report
    
    df, scale_report = detect_scale_inconsistency(df)
    cleaning_report["scale_fixes"] = scale_report
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    missing_before = df[numeric_cols].isna().sum().sum()
    
    if use_midas and midas_model is not None:
        try:
            X = df[numeric_cols].values.astype(np.float32)
            mask = np.isnan(X)
            if mask.any():
                X_imp = midas_model.impute(torch.FloatTensor(np.nan_to_num(X)), torch.BoolTensor(mask))
                df[numeric_cols] = X_imp.numpy()
                cleaning_report["missing_fixes"]["method"] = "midas"
        except:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            cleaning_report["missing_fixes"]["method"] = "median_fallback"
    else:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        cleaning_report["missing_fixes"]["method"] = "median"
    
    missing_after = df[numeric_cols].isna().sum().sum()
    cleaning_report["missing_fixes"]["before"] = int(missing_before)
    cleaning_report["missing_fixes"]["after"] = int(missing_after)
    cleaning_report["missing_fixes"]["fixed"] = int(missing_before - missing_after)
    cleaning_report["final_shape"] = df.shape
    return df, cleaning_report

def detect_foreign_keys(dfs, file_names=None):
    relations = []
    for i, df1 in enumerate(dfs):
        for j, df2 in enumerate(dfs):
            if i >= j:
                continue
            for col1 in df1.columns:
                for col2 in df2.columns:
                    if col1 == col2:
                        continue
                    try:
                        vals1 = set(df1[col1].dropna().astype(str))
                        vals2 = set(df2[col2].dropna().astype(str))
                        if len(vals1) == 0 or len(vals2) == 0:
                            continue
                        overlap = len(vals1 & vals2) / min(len(vals1), len(vals2))
                        if overlap > 0.7:
                            relations.append({
                                "table1": file_names[i] if file_names else f"df{i}",
                                "col1": col1,
                                "table2": file_names[j] if file_names else f"df{j}",
                                "col2": col2,
                                "overlap": overlap
                            })
                    except:
                        continue
    return relations


def detect_time_columns(df):
    time_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(t in col_lower for t in ['date', 'time', 'timestamp', 'datetime']):
            time_cols.append(col)
            continue
        if df[col].dtype in ['int64', 'float64']:
            continue
        try:
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.contains(r'[-/]').sum() < len(sample) * 0.5:
                continue
            pd.to_datetime(sample, errors='raise')
            time_cols.append(col)
        except:
            continue
    return time_cols


def add_time_features(df, time_col):
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[f'{time_col}_year'] = df[time_col].dt.year
        df[f'{time_col}_month'] = df[time_col].dt.month
        df[f'{time_col}_day'] = df[time_col].dt.day
        df[f'{time_col}_dayofweek'] = df[time_col].dt.dayofweek
        df[f'{time_col}_hour'] = df[time_col].dt.hour.fillna(0)
        df[f'{time_col}_quarter'] = df[time_col].dt.quarter
    except:
        pass
    return df


def add_lag_features(df, time_col, value_cols, lags=[1, 3, 7]):
    if time_col not in df.columns:
        return df
    try:
        df = df.sort_values(time_col)
        for col in value_cols[:5]:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            df[f'{col}_rolling3'] = df[col].rolling(3, min_periods=1).mean()
            df[f'{col}_rolling7'] = df[col].rolling(7, min_periods=1).mean()
    except:
        pass
    return df


def smart_time_series_prep(df):
    report = {"time_columns": [], "features_added": 0}
    time_cols = detect_time_columns(df)
    report["time_columns"] = time_cols
    
    if not time_cols:
        return df, report
    
    main_time_col = time_cols[0]
    df = add_time_features(df, main_time_col)
    report["features_added"] += 6
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        original_cols = len(df.columns)
        df = add_lag_features(df, main_time_col, numeric_cols[:5])
        report["features_added"] += len(df.columns) - original_cols
    
    return df, report

def is_time_pattern(series):
    """Değerlerin zaman formatı olup olmadığını kontrol et"""
    sample = series.dropna().head(20).astype(str)
    time_patterns = 0
    for val in sample:
        # HH:MM:SS, HH:MM, YYYY-MM-DD formatları
        if ':' in val and len(val) <= 12:
            time_patterns += 1
        elif '-' in val and len(val) >= 8 and len(val) <= 10:
            time_patterns += 1
    return time_patterns > len(sample) * 0.5

def is_id_pattern(series, n_samples):
    """Değerlerin ID/index pattern'ı olup olmadığını kontrol et"""
    nunique = series.nunique()
    # Çok yüksek unique oranı = muhtemelen ID
    if nunique > n_samples * 0.8:
        return True
    # Sequential sayılar = muhtemelen index
    if series.dtype in ['int64', 'float64']:
        sorted_vals = series.dropna().sort_values().values
        if len(sorted_vals) > 10:
            diffs = np.diff(sorted_vals[:20])
            if len(set(diffs)) <= 2:  # Hep aynı fark = sequential
                return True
    return False


def smart_merge_datasets(dataframes, file_names=None):
    """
    Birden fazla dataseti akıllıca birleştirir.
    - Player ID mapping otomatik tespit
    - Prefix ekleme
    - NaN handling
    """
    import pandas as pd
    import numpy as np
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    if len(dataframes) == 0:
        return None
    
    def find_player_col(df):
        candidates = ['player_id', 'player.id', 'Player Full Name (P)', 'player_num', 'id']
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if 'player' in col.lower() and ('id' in col.lower() or 'name' in col.lower()):
                return col
        return None
    
    def extract_player_num(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        if 'player' in val_str.lower():
            val_str = val_str.replace('Playre', 'Player')
            try:
                return int(val_str.lower().replace('player', '').strip())
            except:
                pass
        try:
            return int(float(val_str))
        except:
            return None
    
    known_mappings = {
        33098:1, 35690:2, 33126:3, 30500:4, 23428:5, 6357:6, 47467:7, 
        43555:8, 30458:9, 3016:10, 276020:11, 36167:12, 37999:13, 
        30944:14, 30736:15, 26400:16,
        34810:1, 25980:2, 36205:3, 36109:4, 18224:5, 5247:6, 31954:7,
        36103:8, 34819:9, 616:10, 502999:11, 34766:12, 38250:13,
        18289:14, 25463:15, 18024:16
    }
    
    prepared_dfs = []
    
    for i, df in enumerate(dataframes):
        prefix = f'd{i}' if file_names is None else file_names[i].split('.')[0][:10]
        prefix = prefix.replace(' ', '_').replace('-', '_')
        
        player_col = find_player_col(df)
        df = df.copy()
        
        if player_col and player_col != "player_num":
            if player_col == 'Player Full Name (P)':
                df['player_num'] = df[player_col].apply(extract_player_num)
            elif df[player_col].dtype in ['int64', 'float64']:
                df['player_num'] = df[player_col].map(known_mappings)
                if df['player_num'].isna().all():
                    df['player_num'] = df[player_col].apply(extract_player_num)
            else:
                df['player_num'] = df[player_col].apply(extract_player_num)
            
            df = df[df['player_num'].notna()].copy()
        
        if len(df) == 0:
            continue
        
        new_cols = {}
        for col in df.columns:
            if col == 'player_num':
                new_cols[col] = col
            else:
                new_cols[col] = f'{prefix}_{col}'
        
        df = df.rename(columns=new_cols)
        prepared_dfs.append(df)
    
    if len(prepared_dfs) == 0:
        return dataframes[0] if len(dataframes) > 0 else None
    
    if len(prepared_dfs) == 1:
        return prepared_dfs[0]
    
    def find_best_merge_key(dfs):
        """Tüm df'lerde ortak olan kolonları bul, en iyi merge key'i seç"""
        # Tüm df'lerde ortak kolonlar
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        
        if not common_cols:
            return None
        
        # Her ortak kolon için unique ratio hesapla - en iyi key yüksek unique, düşük null
        best_key = None
        best_score = -1
        
        for col in common_cols:
            try:
                scores = []
                for df in dfs:
                    if col not in df.columns:
                        continue
                    total = len(df)
                    if total == 0:
                        continue
                    unique = df[col].nunique()
                    non_null = df[col].notna().sum()
                    # Score: unique ratio * non-null ratio
                    score = (unique / total) * (non_null / total)
                    scores.append(score)
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    # Prefer columns with 'id', 'key', 'num' in name
                    col_lower = col.lower()
                    if 'id' in col_lower or 'key' in col_lower or 'num' in col_lower:
                        avg_score *= 1.2
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_key = col
            except:
                continue
        
        # Minimum threshold - key en az %10 unique olmalı
        if best_score < 0.1:
            return None
        
        return best_key
    
    fk_relations = detect_foreign_keys(prepared_dfs, file_names)
    if fk_relations:
        print(f"Detected foreign keys: {fk_relations}")
    merge_key = find_best_merge_key(prepared_dfs)
    
    if merge_key:
        sorted_dfs = sorted(prepared_dfs, key=len, reverse=True)
        merged = sorted_dfs[0]
        
        for df in sorted_dfs[1:]:
            if len(df) > len(merged) * 0.5:
                df_agg = df.groupby(merge_key).mean(numeric_only=True).reset_index()
                for col in df.columns:
                    if col != merge_key and col not in df_agg.columns:
                        df_agg[col] = df.groupby(merge_key)[col].first().values
                merged = pd.merge(merged, df_agg, on=merge_key, how='left')
            else:
                df_agg = df.groupby(merge_key).mean(numeric_only=True).reset_index()
                merged = pd.merge(merged, df_agg, on=merge_key, how='left')
        merged = merged.fillna(0)
        print(f"DEBUG MERGE: {len(dataframes)} files -> {merged.shape}")
    else:
        # No good key - check row counts
        row_counts = [len(df) for df in prepared_dfs]
        if len(set(row_counts)) == 1:
            # Same rows - column concat
            merged = pd.concat(prepared_dfs, axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated()]
            merged = merged.fillna(0)
            print(f"Smart merge (col-concat): {len(dataframes)} datasets -> {merged.shape}")
        else:
            # Row concat fallback
            merged = pd.concat(prepared_dfs, axis=0, ignore_index=True)
            merged = merged.fillna(0)
            print(f"Smart merge (row-concat): {len(dataframes)} datasets -> {merged.shape}")
    
    return merged


def get_dynamic_config(n_samples, n_features, n_classes):
    import numpy as np
    
    complexity = (n_samples * n_classes) / 1000
    
    if complexity > 5000:
        d_model = 512
    elif complexity > 200:
        d_model = 256
    else:
        d_model = 128
    
    if n_samples > 50000:
        n_layers = 3
    else:
        n_layers = 2
    
    if complexity > 1000:
        n_latents = 128
    elif complexity > 100:
        n_latents = 64
    else:
        n_latents = 32
    
    max_cols = max(64, int(np.ceil(n_features / 32) * 32))
    
    if n_samples < 100:
        batch_size = 4
    elif n_samples < 500:
        batch_size = 16
    elif n_samples < 2000:
        batch_size = 32
    else:
        batch_size = 64
    
    if n_features > 150:
        batch_size = max(batch_size, 128)
    elif n_features > 100:
        batch_size = max(batch_size, 64)
    
    if n_samples < 100:
        epochs = 100
    elif n_samples < 500:
        epochs = 50
    elif n_samples < 2000:
        epochs = 30
    else:
        epochs = 20
    
    if n_classes > 20:
        epochs = min(epochs + 5, 100)
    
    patience = max(5, min(25, epochs // 4))
    
    if batch_size <= 8:
        lr = 0.0005
    elif batch_size <= 32:
        lr = 0.001
    else:
        lr = 0.002
    
    n_heads = max(4, d_model // 64)
    
    return {
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'n_latents': n_latents,
        'n_features': n_features,
        'n_classes': n_classes,
        'n_sectors': min(n_classes, 10),
        'max_cols': max_cols,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
        'lr': lr
    }



def advanced_target_score(df, col):
    """
    %100 DİNAMİK Target Skorlama
    
    SIFIR hardcoded keyword/pattern
    Sadece veri analizi ile karar verir
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    series = df[col]
    n_samples = len(df)
    nunique = series.nunique()
    
    # === TEMEL FİLTRELER (Tamamen Dinamik) ===
    
    # 1. Numeric kolonlar target olamaz
    if series.dtype in ['float64', 'float32', 'int64', 'int32', 'float', 'int']:
        return None
    
    # 2. Minimum 2 class
    if nunique < 2:
        return None
    
    # 3. ID Tespiti - unique oranı ve mutlak sayı
    unique_ratio = nunique / n_samples
    if unique_ratio > 0.5:
        return None
    
    # Çok fazla class = muhtemelen ID/name kolonu
    if nunique > 500:
        return None
    
    # 4. Neredeyse sabit kolon
    value_counts = series.value_counts()
    if value_counts.iloc[0] / n_samples > 0.98:
        return None
    
    # 5. Değer Analizi - string özellikleri
    try:
        sample_vals = series.dropna().head(200).astype(str)
        
        # 5a. Ortalama uzunluk - çok uzun = text/description
        avg_len = np.mean([len(v) for v in sample_vals])
        if avg_len > 50:
            return None
        
        # 5b. Kelime sayısı - çok fazla kelime = text
        avg_words = np.mean([len(v.split()) for v in sample_vals])
        if avg_words > 5:
            return None
        
        # 5c. Sayı oranı yüksek + özel karakter = ID/code
        digit_ratios = [sum(c.isdigit() for c in v) / max(len(v), 1) for v in sample_vals]
        avg_digit_ratio = np.mean(digit_ratios)
        
        special_ratios = [sum(c in '-_/:@#.' for c in v) / max(len(v), 1) for v in sample_vals]
        avg_special = np.mean(special_ratios)
        
        # Çok fazla rakam + özel karakter = muhtemelen ID/tarih/kod
        if avg_digit_ratio > 0.5 and avg_special > 0.1:
            return None
        
        # 5d. Tüm değerler unique uzunlukta ve uzun = hash/token
        lengths = [len(v) for v in sample_vals]
        if len(set(lengths)) < 3 and np.mean(lengths) > 20:
            return None
            
    except:
        pass
    
    # === PREDICTABILITY (En Önemli Metrik) ===
    predictability = 0.3  # Default düşük
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) >= 2:
            sample_size = min(2000, n_samples)
            X_sample = df[num_cols].fillna(0).values[:sample_size]
            
            le = LabelEncoder()
            y_sample = le.fit_transform(series.fillna('__NA__').astype(str)[:sample_size])
            
            # Eğer bir class çok baskınsa RF yanıltıcı olabilir
            class_counts = np.bincount(y_sample)
            max_class_ratio = class_counts.max() / len(y_sample)
            
            # Her zaman LIFT hesapla (RF - majority baseline)
            rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42, n_jobs=-1)
            cv = 2 if sample_size < 500 else 3
            scores = cross_val_score(rf, X_sample, y_sample, cv=cv, scoring='accuracy')
            rf_acc = scores.mean()
            
            # Majority baseline
            baseline = max_class_ratio
            
            # LIFT = RF accuracy - baseline
            # Negatif lift = kötü target (model baseline'dan iyi değil)
            lift = rf_acc - baseline
            
            # Predictability = normalized lift (0-1 arası)
            # lift > 0.1 iyi, lift < 0 kötü
            if lift > 0:
                predictability = 0.5 + (lift * 2)  # Pozitif lift bonus
                predictability = min(1.0, predictability)
            else:
                predictability = 0.3 + lift  # Negatif lift penalty
                predictability = max(0.1, predictability)
    except:
        pass
    
    # === DİĞER METRİKLER ===
    
    # Entropy - dağılım dengesi
    probs = value_counts / value_counts.sum()
    entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
    max_entropy = np.log2(nunique) if nunique > 1 else 1
    entropy_score = entropy / max_entropy if max_entropy > 0 else 0
    
    # Imbalance
    imbalance_ratio = value_counts.max() / (value_counts.min() + 1)
    imbalance_score = 1 / (1 + np.log10(imbalance_ratio + 1))
    
    # Missing
    missing_ratio = series.isna().sum() / n_samples
    missing_score = 1 - missing_ratio
    
    # Class count score - 3+ class GÜÇLÜ tercih
    if nunique == 2:
        class_score = 0.3  # Binary çok kolay, az bilgi içerir
    elif 3 <= nunique <= 100:
        class_score = 1.0
    else:
        class_score = 0.7
    
    # === FİNAL SKOR ===
    final_score = (
        predictability * 0.45 +
        entropy_score * 0.20 +
        imbalance_score * 0.15 +
        class_score * 0.15 +      # Class count daha önemli
        missing_score * 0.05
    ) * 100
    
    return final_score


def auto_select_target(df, user_target=None):
    """
    %100 DİNAMİK Target Seçimi
    Hardcoded değer YOK
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    if user_target and user_target in df.columns:
        return user_target
    
    candidates = []
    
    for col in df.columns:
        score = advanced_target_score(df, col)
        if score is not None:
            candidates.append((col, score))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Target candidates: {[(c[0], round(c[1], 1)) for c in candidates[:5]]}")
        return candidates[0][0]
    
    # Fallback - en az unique kategorik
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        return min(cat_cols, key=lambda c: df[c].nunique())
    
    return df.columns[-1]


def smart_column_mapping(df_cols, target_col):
    """Agnostik kolon mapping - tüm sayısal kolonları feature olarak kullan"""
    feature_cols = [c for c in df_cols if c != target_col]
    # Mapping: her kolon kendine map'lenir
    mapped = {col: col for col in feature_cols}
    return mapped, feature_cols


app = Flask(__name__)
app.json_encoder = NaNSafeEncoder


class MIDAS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 10)
        )
    def forward(self, x, mask):
        return self.decoder(self.encoder(torch.cat([x * mask, mask], dim=1)))
    def impute(self, x, mask, n_iter=3):
        current = x * mask
        for _ in range(n_iter):
            current = x * mask + self.forward(current, mask) * (1 - mask)
        return current


# TabularFoundationModel - her CSV için yeni model
SERVER_PORT = int(os.getenv("FLASK_PORT", 6000))
# print("=" * 60)
# print("SCHEMALABSAI - TabularFoundationModel Server")
# print("=" * 60)
current_model_name = "tabular_foundation"
finetuned_models = {}


# print(f"Model: TabularFoundationModel v2.1")
# print(f"Server ready on port {SERVER_PORT}")
# print("=" * 60)

training_sessions = {}
training_progress = {"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "idle", "eta": "0%", "start_time": 0}

# Session'ları dosyaya kaydet - Flask restart'ta kaybolmasın
SESSIONS_FILE = '/tmp/schemalabs_training_sessions.json'

def _load_sessions():
    global training_sessions
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'r') as f:
                training_sessions = json.load(f)
    except:
        training_sessions = {}

def _save_sessions():
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(training_sessions, f)
    except:
        pass

def get_session(query_id):
    _load_sessions()
    if query_id not in training_sessions:
        training_sessions[query_id] = {"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "idle", "eta": "0%", "start_time": 0, "query_id": query_id}
        _save_sessions()
    return training_sessions[query_id]

def save_session(query_id, session):
    training_sessions[query_id] = session
    _save_sessions()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_type": "TabularFoundationModel",
        "version": "2.1",
        "finetuned_models": list(finetuned_models.keys())
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": "TabularFoundationModel",
        "version": "2.1",
        "finetuned_models": len(finetuned_models),
        "capabilities": ["classification", "sector_detection", "midas_imputation", "ewc_learning"]
    })

@app.route('/sectors', methods=['GET'])
def list_sectors():
    return jsonify({
        "info": "Sectors auto-detected during fine-tuning",
        "detection": "dynamic",
        "finetuned_models": list(finetuned_models.keys())
    })

@app.route('/models/list', methods=['GET'])
def list_models():
    models = []
    checkpoint_dir = Path('../checkpoints')
    for f in sorted(checkpoint_dir.glob('*.pt'), reverse=True):
        filename = f.name
        models.append({
            "name": filename.replace('.pt', ''),
            "filename": filename,
            "path": str(f),
            "type": "finetuned" if "finetuned" in filename else "base",
            "is_current": filename == "schemalabsai_v1.pt"
        })
    return jsonify({"models": models, "current": current_model_name})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sector and subsector from raw values"""
    try:
        data = request.json
        values = np.array(data['values'], dtype=np.float32)
        sector_hint = data.get('sector', None)
        
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        if values.shape[1] < 10:
            pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
            values = np.hstack([values, pad])
        elif values.shape[1] > 10:
            values = values[:, :10]
        
        results = []
        
        for i in range(len(values)):
            row = values[i:i+1]
            
            best_sector = None
            if sector_hint and sector_hint in sector_bases:
                best_sector = sector_hint
            
            if best_sector:
                base = np.array(sector_bases[best_sector])
                row_sub = row - base
            else:
                row_sub = row
            
            row_norm = (row_sub - X_min) / (X_max - X_min + 1e-8)
            
            mask = ~np.isnan(row_norm)
            row_norm = np.nan_to_num(row_norm, nan=0.0)
            
            X_t = torch.FloatTensor(row_norm)
            mask_t = torch.FloatTensor(mask.astype(np.float32))
            
            with torch.inference_mode():
                if mask.mean() < 1.0:
                    X_imp = midas.impute(X_t, mask_t)
                else:
                    X_imp = X_t
                
                try:
                    out = model(X_imp)
                    sec_logits = out.get('sector', torch.zeros(X_imp.shape[0], 50))
                except:
                    sec_logits = torch.zeros(X_imp.shape[0], 50)
                sec_probs = F.softmax(sec_logits, dim=1)
                sec_conf, sec_pred = sec_probs.max(1)
                
                sub_logits = sec_logits  # Use sector logits as subsector
                sub_probs = F.softmax(sub_logits, dim=1)
                sub_conf, sub_pred = sub_probs.max(1)
            
            sid = sec_pred.item()
            sub_id = sub_pred.item()
            
            sector_name = id_to_sector.get(sid, f"sector_{sid}")
            subsector_name = id_to_subsector.get(sid, {}).get(sub_id, f"subsector_{sub_id}")
            
            results.append({
                "sector": sector_name,
                "sector_id": sid,
                "sector_confidence": float(sec_conf.item()),
                "subsector": subsector_name,
                "subsector_id": sub_id,
                "subsector_confidence": float(sub_conf.item()),
                "combined_confidence": float(sec_conf.item() * sub_conf.item())
            })
        
        return jsonify({
            "predictions": results,
            "model_used": current_model_name,
            "status": "success"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction for CSV data"""
    try:
        data = request.json
        values = np.array(data['values'], dtype=np.float32)
        
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        if values.shape[1] < 10:
            pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
            values = np.hstack([values, pad])
        elif values.shape[1] > 10:
            values = values[:, :10]
        
        sector_hint = data.get('sector', None)
        
        if sector_hint and sector_hint in sector_bases:
            base = np.array(sector_bases[sector_hint])
            values = values - base
        
        values_norm = (values - X_min) / (X_max - X_min + 1e-8)
        
        X_t = torch.FloatTensor(values_norm)
        with torch.inference_mode():
            try:
                out = model(X_t)
                sec_logits = out.get('sector', None)
                if sec_logits is None:
                    sec_logits = torch.zeros(X_t.shape[0], 50)
            except:
                sec_logits = torch.zeros(X_t.shape[0], 50)
            sec_probs = F.softmax(sec_logits, dim=1)
            sec_conf, sec_pred = sec_probs.max(1)
            
            sub_logits = sec_logits
            sub_probs = F.softmax(sub_logits, dim=1)
            sub_conf, sub_pred = sub_probs.max(1)
        
        predictions = []
        for i in range(len(values)):
            sid = sec_pred[i].item()
            sub_id = sub_pred[i].item()
            predictions.append({
                "sector": id_to_sector.get(sid, f"sector_{sid}"),
                "subsector": id_to_subsector.get(sid, {}).get(sub_id, f"subsector_{sub_id}"),
                "sector_confidence": float(sec_conf[i].item()),
                "subsector_confidence": float(sub_conf[i].item())
            })
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "model_used": current_model_name,
            "status": "success"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Smart analyzer - query-aware, token-efficient"""
    try:
        data = request.json
        file_id = data.get('file_id', '')
        query = data.get('query', data.get('message', '')).lower()
        
        uploads_dir = '../uploads'
        file_path = None
        if os.path.exists(uploads_dir):
            for f in os.listdir(uploads_dir):
                if len(file_id) >= 8 and f.startswith(file_id[:8]):
                    file_path = os.path.join(uploads_dir, f)
                    break
        
        if not file_path:
            return jsonify({'analysis': 'File not found.', 'status': 'error'})
        
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        
        # === ADVANCED ANALYTICS ENGINE ===
        detected_types = detect_analytics_type(query)
        if detected_types:
            print(f"Detected analytics types: {[d['type'] for d in detected_types]}")
            advanced_analysis = generate_analytics(df, query, detected_types)
            if advanced_analysis and len(advanced_analysis) > 100:
                return jsonify({'analysis': advanced_analysis, 'status': 'success'})
        
        # === COMPACT ANALYSIS (max 8K chars) ===
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        query_words = [w.lower() for w in query.replace('?', '').replace(',', '').split() if len(w) > 2]
        # Add common synonyms
        expanded_words = list(query_words)
        for w in query_words:
            if w == 'distance': expanded_words.append('distance_covered')
            if w == 'speed': expanded_words.append('speed_avg')
            if w == 'player': expanded_words.extend(['player_in_possession_name', 'player_name'])
        query_words = expanded_words
        
        # Smart match: find columns matching query words
        matched_num = []
        matched_cat = []
        
        # Score columns by how well they match query
        for col in num_cols:
            col_lower = col.lower()
            score = 0
            for w in query_words:
                if w in col_lower:
                    score += len(w)  # Longer match = higher score
                    if col_lower == w or col_lower.endswith('_' + w) or col_lower.startswith(w + '_'):
                        score += 10  # Exact/partial match bonus
            if score > 0:
                matched_num.append((col, score))
        
        for col in cat_cols:
            col_lower = col.lower()
            score = 0
            for w in query_words:
                if w in col_lower:
                    score += len(w)
                    if 'name' in col_lower:
                        score += 20  # Name columns are better for grouping
            if score > 0:
                matched_cat.append((col, score))
        
        # Sort by score descending
        matched_num = [c[0] for c in sorted(matched_num, key=lambda x: -x[1])]
        matched_cat = [c[0] for c in sorted(matched_cat, key=lambda x: -x[1])]
        
        analysis = f"DATASET: {len(df)} rows, {len(df.columns)} columns\n"
        analysis += f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}\n\n"
        
        # === ADVANCED ANALYTICS DETECTION ===
        is_swot = any(w in query for w in ['swot', 'strength', 'weakness', 'opportunity', 'threat'])
        is_benchmark = any(w in query for w in ['benchmark', 'compare to average', 'vs average', 'team average'])
        is_risk = any(w in query for w in ['risk', 'injury', 'danger', 'concern'])
        is_trend = any(w in query for w in ['trend', 'over time', 'progression'])
        is_anomaly = any(w in query for w in ['outlier', 'anomaly', 'unusual', 'exceptional'])
        
        # Extract player name from query if mentioned
        player_name = None
        for cat_col in cat_cols:
            if 'name' in cat_col.lower() or 'player' in cat_col.lower():
                for val in df[cat_col].dropna().unique():
                    val_str = str(val).lower()
                    if len(val_str) > 3 and val_str in query:
                        player_name = val
                        break
                if not player_name:
                    # Try partial match
                    query_parts = query.split()
                    for val in df[cat_col].dropna().unique():
                        val_parts = str(val).lower().split()
                        if any(vp in query for vp in val_parts if len(vp) > 3):
                            player_name = val
                            break
        
        if player_name and (is_swot or is_benchmark):
            analysis += f"=== PLAYER PROFILE: {player_name} ===\n"
            for cat_col in cat_cols:
                if 'name' in cat_col.lower() or 'player' in cat_col.lower():
                    player_data = df[df[cat_col] == player_name]
                    if len(player_data) > 0:
                        analysis += f"Records: {len(player_data)}\n"
                        for num_col in num_cols[:20]:
                            try:
                                player_val = player_data[num_col].mean()
                                team_avg = df[num_col].mean()
                                team_std = df[num_col].std()
                                diff_pct = ((player_val - team_avg) / team_avg * 100) if team_avg != 0 else 0
                                status = "ABOVE" if player_val > team_avg else "BELOW"
                                analysis += f"  {num_col}: {player_val:.2f} (Team avg: {team_avg:.2f}, {status} by {abs(diff_pct):.1f}%)\n"
                            except:
                                pass
                        break
            analysis += "\n"
        
        if is_risk and matched_num:
            analysis += "=== RISK INDICATORS ===\n"
            for num_col in matched_num[:5]:
                try:
                    mean_val = df[num_col].mean()
                    std_val = df[num_col].std()
                    high_threshold = mean_val + 2 * std_val
                    high_risk = df[df[num_col] > high_threshold]
                    analysis += f"{num_col}: Mean={mean_val:.2f}, Std={std_val:.2f}, High risk threshold={high_threshold:.2f}, Count above={len(high_risk)}\n"
                except:
                    pass
            analysis += "\n"
        
        if is_anomaly and matched_num:
            analysis += "=== OUTLIERS (>2 std dev) ===\n"
            for num_col in matched_num[:3]:
                try:
                    mean_val = df[num_col].mean()
                    std_val = df[num_col].std()
                    outliers = df[(df[num_col] > mean_val + 2*std_val) | (df[num_col] < mean_val - 2*std_val)]
                    if len(outliers) > 0 and matched_cat:
                        for cat_col in matched_cat[:1]:
                            analysis += f"{num_col} outliers by {cat_col}:\n"
                            for _, row in outliers.head(10).iterrows():
                                analysis += f"  {row[cat_col]}: {row[num_col]:.2f}\n"
                except:
                    pass
            analysis += "\n"
        
        # AUTO-CALCULATE if matching columns found
        if matched_num and matched_cat:
            analysis += "=== QUERY RESULT ===\n"
            for num_col in matched_num[:2]:
                for cat_col in matched_cat[:1]:
                    try:
                        if any(w in query for w in ['average', 'avg', 'mean']):
                            result = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                            agg = "AVG"
                        elif any(w in query for w in ['count', 'how many']):
                            result = df.groupby(cat_col)[num_col].count().sort_values(ascending=False)
                            agg = "COUNT"
                        elif any(w in query for w in ['max', 'highest']):
                            result = df.groupby(cat_col)[num_col].max().sort_values(ascending=False)
                            agg = "MAX"
                        elif any(w in query for w in ['min', 'lowest']):
                            result = df.groupby(cat_col)[num_col].min().sort_values(ascending=True)
                            agg = "MIN"
                        else:
                            result = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                            agg = "TOTAL"
                        analysis += f"\n{agg} {num_col} BY {cat_col}:\n"
                        for idx, val in list(result.items())[:25]:
                            analysis += f"  {idx}: {val:.2f}\n"
                    except:
                        pass
            analysis += "\n"
        elif matched_num:
            analysis += "=== MATCHED METRICS ===\n"
            for col in matched_num[:3]:
                analysis += f"{col}: sum={df[col].sum():.2f}, avg={df[col].mean():.2f}\n"
            analysis += "\n"

        
        # ALL numeric column names listed, stats for first 30
        analysis += "NUMERIC COLUMNS:\n"
        for i, col in enumerate(num_cols):
            try:
                if i < 30:
                    analysis += f"  {col}: avg={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}\n"
                else:
                    # Just list the name
                    pass
            except:
                pass
        # List ALL remaining column names compactly
        if len(num_cols) > 30:
            remaining = num_cols[30:]
            analysis += f"  MORE NUMERIC ({len(remaining)}): {', '.join(remaining)}\n"
        
        # Categorical columns with values (max 10)
        analysis += "\nCATEGORICAL COLUMNS:\n"
        for col in cat_cols[:10]:
            nunique = df[col].nunique()
            if nunique <= 15:
                vals = df[col].dropna().unique().tolist()
                analysis += f"  {col}: {vals}\n"
            else:
                top = df[col].value_counts().head(5).index.tolist()
                analysis += f"  {col} ({nunique} unique): {top}...\n"
        if len(cat_cols) > 10:
            analysis += f"  ... +{len(cat_cols)-10} more categorical columns\n"
        
        # Sample rows (5 rows, max 8 columns)
        show_cols = (cat_cols[:2] + num_cols[:6])[:8]
        analysis += f"\nSAMPLE DATA ({len(show_cols)} cols):\n"
        analysis += " | ".join([c[:15] for c in show_cols]) + "\n"
        for _, row in df.head(5).iterrows():
            vals = [str(row[c])[:12] for c in show_cols]
            analysis += " | ".join(vals) + "\n"
        
        # Truncate if still too long
        if len(analysis) > 8000:
            analysis = analysis[:8000] + "\n...(truncated)"
        
        return jsonify({'analysis': analysis, 'status': 'success'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'analysis': f'Error: {e}', 'status': 'error'})

@app.route('/finetune', methods=['POST'])
def finetune():
    """Fine-tune model on user data"""
    try:
        epochs_req = int(request.form.get('epochs', 0))  # 0 = auto
        batch_size_req = int(request.form.get('batch_size', 0))  # 0 = auto
        target_column = request.form.get('target_column', None)
        query_id = request.form.get('query_id', 'default')
        analyze_only = request.form.get('analyze_only', 'false').lower() == 'true'
        
        session = get_session(query_id)
        print(f"DEBUG FINETUNE START: query_id={query_id}, epochs_req={epochs_req}, analyze_only={analyze_only}")
        # Reset session for new training
        session.update({"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "starting", "eta": "0%", "start_time": time.time(), "query_id": query_id})
        
        merge_files = request.form.get('merge_files', 'false').lower() == 'true'
        
        # Çoklu dosya kontrolü
        files = request.files.getlist('file')
        if not files or len(files) == 0:
            if 'file' in request.files:
                files = [request.files['file']]
            else:
                return jsonify({"error": "No file provided"}), 400
        
        dataframes = []
        file_names = []
        
        for file in files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            file.save(temp_file.name)
            temp_file.close()
            
            # CSV, Excel, JSON, Parquet
            if file.filename.endswith(('.xlsx', '.xls')):
                df_temp = pd.read_excel(temp_file.name)
            elif file.filename.endswith('.json'):
                df_temp = pd.read_json(temp_file.name)
            elif file.filename.endswith('.parquet'):
                df_temp = pd.read_parquet(temp_file.name)
            else:
                df_temp = pd.read_csv(temp_file.name)
            
            dataframes.append(df_temp)
            file_names.append(file.filename)
            os.unlink(temp_file.name)
        
        # Birden fazla dosya varsa smart merge yap
        merged_file_id = None
        if len(dataframes) > 1 and merge_files:
            print(f"Smart merging {len(dataframes)} files: {file_names}")
            df = smart_merge_datasets(dataframes, file_names)
            print(f"Merged shape: {df.shape}")
            
            # Save merged file to uploads
            import uuid
            from datetime import datetime
            merged_file_id = str(uuid.uuid4())
            from datetime import datetime; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_filename = f"{merged_file_id[:8]}_merged_all_{timestamp}.csv"
            merged_path = os.path.join('../uploads', merged_filename)
            df.to_csv(merged_path, index=False)
            print(f"Merged file saved: {merged_path}")
        else:
            df = dataframes[0]
        
        # Otomatik akıllı target seçimi
        target_col = auto_select_target(df, target_column)
        print(f"Auto-selected target: {target_col}")
        
        df, cleaning_report = smart_data_cleaning(df)
        df, ts_report = smart_time_series_prep(df)
        print(f"Time series prep: {ts_report}")
        print(f"Data cleaning: {cleaning_report}")
        numeric_df = df.select_dtypes(include=['number'])
        # Agnostik: Tüm sayısal kolonları feature olarak kullan
        col_mapping, feature_cols = smart_column_mapping(numeric_df.columns.tolist(), target_col)
        
        print(f"Feature columns: {feature_cols}")
        
        # Tüm feature kolonlarını al
        X = df[feature_cols].values.astype(np.float32)
        
        # Eksik değer kontrolü
        has_missing = np.isnan(X).any()
        if has_missing:
            missing_pct = np.isnan(X).mean() * 100
            X = np.nan_to_num(X, nan=0.0)
            print(f"Missing data filled: {missing_pct:.1f}%")
        
        numeric_cols = feature_cols
        
        le = LabelEncoder()
        # Mixed type fix - hepsini string yap
        y = le.fit_transform(df[target_col].astype(str))
        n_classes = len(le.classes_)
        
        if analyze_only:
            n_samples = len(df)
            smart_epochs = min(20, max(5, n_samples // 1000))
            smart_batch = min(128, max(32, n_samples // 100))
            return jsonify({
                "n_samples": n_samples,
                "n_classes": n_classes,
                "n_features": X.shape[1] if 'X' in dir() else len(numeric_df.columns),
                "target_column": target_col,
            "miras_enabled": use_miras if 'use_miras' in dir() else False,
                "smart_epochs": smart_epochs,
                "smart_batch_size": smart_batch,
                "classes": le.classes_.tolist()
            })
        
        
        # StandardScaler ile normalize et - her türlü veri range'i için
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        
        # TabularFoundationModel herhangi feature sayısı ile çalışır
        # Agnostik: Herhangi feature sayısı ile çalışır
        input_dim = X.shape[1]
        print(f"Training with {input_dim} features")
        
        # Tam dinamik config
        dyn_cfg = get_dynamic_config(len(X), input_dim, n_classes)
        
        ft_config = {
            'd_model': dyn_cfg['d_model'],
            'n_heads': dyn_cfg['n_heads'],
            'n_layers': dyn_cfg['n_layers'],
            'schema_layers': dyn_cfg['n_layers'],
            'n_latents': dyn_cfg['n_latents'],
            'n_features': input_dim,
            'n_classes': n_classes,
            'vocab_size': 50000,
            'n_types': 10,
            'max_cols': dyn_cfg['max_cols']
        }
        
        print(f"Dynamic: d={dyn_cfg['d_model']}, L={dyn_cfg['n_layers']}, lat={dyn_cfg['n_latents']}, bs={dyn_cfg['batch_size']}, ep={dyn_cfg['epochs']}, lr={dyn_cfg['lr']:.4f}")
        # Check if MIRAS is requested
        use_miras = request.form.get('use_miras', 'false').lower() == 'true'
        miras_bias = request.form.get('miras_bias', 'huber')
        miras_retention = request.form.get('miras_retention', 'lq')
        
        print(f"Creating model with config: {ft_config}, MIRAS={use_miras}")
        try:
            if use_miras:
                miras_config = {
                    'attentional_bias': miras_bias,
                    'retention_gate': miras_retention,
                    'p': 3.0, 'q': 4.0, 'delta': 1.0,
                    'use_momentum': True,
                    'use_channel_wise': True,
                    'use_gated_output': True
                }
                ft_model = TabularFoundationModelMIRAS(ft_config, miras_config)
                print(f"MIRAS Model created with bias={miras_bias}, retention={miras_retention}")
            else:
                ft_model = TabularFoundationModel(ft_config)
            print(f"Model created successfully, params: {sum(p.numel() for p in ft_model.parameters())}")
            for m in ft_model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.momentum = 0.01
            if hasattr(torch, "compile") and torch.cuda.is_available():
                try:
                    ft_model = torch.compile(ft_model, mode="default", backend="eager")
                    print("Model compiled for CPU optimization")
                except:
                    pass
                try:
                    ft_model = torch.quantization.quantize_dynamic(ft_model, {nn.Linear}, dtype=torch.qint8)
                    print("Model quantized to INT8")
                except:
                    pass
        except Exception as e:
            print(f"ERROR creating model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        except Exception as e:
            print(f"ERROR creating model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Dinamik batch size ve epochs - algoritmik
        batch_size = batch_size_req if batch_size_req > 0 else dyn_cfg['batch_size']
        epochs = epochs_req if epochs_req > 0 else dyn_cfg['epochs']
        
        optimizer = AdamW(ft_model.parameters(), lr=dyn_cfg['lr'], weight_decay=0.01)
        warmup_epochs = min(3, epochs // 5)
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        loss_fn = nn.CrossEntropyLoss()
        
        # AMP for 2x speedup
        use_amp = torch.cuda.is_available()
        scaler = GradScaler() if use_amp else None
        gradient_accumulation_steps = 2  # Effective batch size *= 2
        
        session["status"] = "training"
        # Aynı progress'i Redis'e de yaz (async için)
        try:
            from async_training import redis_client
            import json
            redis_client.setex(f"training:{query_id}", 3600, json.dumps(session))
        except: pass
        session["start_time"] = time.time()
        session["epochs"] = epochs
        session["epoch"] = 0
        session["accuracy"] = 0.0
        session["loss"] = 0.0
        session["eta"] = "calculating..."
        training_progress.update(session); save_session(query_id, session) if "query_id" in dir() and query_id else training_progress
        
        # Süre optimizasyonu - epoch başına max sample
        if len(X) > 10000:
            max_samples_per_epoch = 10000
        else:
            max_samples_per_epoch = len(X)
        print(f"Starting training loop: X.shape={X.shape}, epochs={epochs}, batch_size={batch_size}, samples_per_epoch={max_samples_per_epoch}")
        
        best_acc = 0
        best_state = None
        patience = dyn_cfg['patience']
        no_improve = 0
        max_epochs = 500  # Maksimum epoch limiti
        current_epoch = 0
        
        # DataLoader ile paralel data loading
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(torch.FloatTensor(X[:max_samples_per_epoch]), torch.LongTensor(y[:max_samples_per_epoch]))
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_mem = torch.cuda.is_available()
        prefetch = 2 if num_workers > 0 else None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, prefetch_factor=prefetch)
        
        while current_epoch < max_epochs:
            print(f"Epoch {current_epoch + 1} starting...")
            ft_model.train()
            total_loss = 0
            correct = 0
            batches = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                
                if np.random.random() > 0.5:
                    noise = torch.randn_like(batch_X) * 0.01
                    batch_X = batch_X + noise
                    batch_X = batch_X.contiguous()
                
                
                # Forward pass
                if use_amp:
                    with autocast():
                        out = ft_model(batch_X)
                        logits = out['output']
                        sector_logits = out['sector']
                        sector_targets = batch_y % ft_model.n_sectors
                        sector_loss = nn.CrossEntropyLoss()(sector_logits, sector_targets)
                        mcm_loss = out.get('mcm_loss', torch.tensor(0.0))
                        miras_loss = out.get('miras_loss', torch.tensor(0.0))
                        loss = loss_fn(logits, batch_y) + sector_loss + 0.1 * mcm_loss + 0.05 * miras_loss
                        loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 0.5 if not torch.cuda.is_available() else 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    out = ft_model(batch_X)
                    logits = out['output']
                    sector_logits = out['sector']
                    sector_targets = batch_y % ft_model.n_sectors
                    sector_loss = nn.CrossEntropyLoss()(sector_logits, sector_targets)
                    mcm_loss = out.get('mcm_loss', torch.tensor(0.0))
                    miras_loss = out.get('miras_loss', torch.tensor(0.0))
                    loss = loss_fn(logits, batch_y) + sector_loss + 0.1 * mcm_loss + 0.05 * miras_loss
                    loss = loss / gradient_accumulation_steps
                    if True:
                        loss.backward()
                    else:
                        continue
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 0.5 if not torch.cuda.is_available() else 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item()
                correct += (logits.argmax(1) == batch_y).sum().item()
                batches += 1
            
            current_epoch += 1
            scheduler.step()
            acc = 100 * correct / min(len(X), max_samples_per_epoch)
            avg_loss = total_loss / max(batches, 1)
            
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in ft_model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            elapsed = time.time() - session["start_time"]
            if current_epoch >= 1:
                time_per_epoch = elapsed / max(current_epoch, 1)
                if best_acc < 95:
                    est_remaining = max(20, epochs - current_epoch)
                else:
                    est_remaining = min(10, max_epochs - current_epoch)
                remaining = est_remaining * time_per_epoch
                eta = f"{int(remaining)}s" if remaining < 60 else f"{int(remaining/60)}m"
            else:
                eta = "..."
            
            session["epoch"] = current_epoch
            # Early stop olacaksa epochs'u güncelle
            if best_acc >= 99.0:
                session["epochs"] = current_epoch
            elif best_acc >= 99.0 and no_improve >= patience:  # %99 hedef
                session["epochs"] = current_epoch
            else:
                session["epochs"] = current_epoch + 1
            session["accuracy"] = acc
            session["loss"] = avg_loss
            session["eta"] = eta
            training_progress.update(session); save_session(query_id, session) if "query_id" in dir() and query_id else training_progress
            
            print(f"Epoch {current_epoch}: Acc={acc:.1f}% (best={best_acc:.1f}%)")
            
            if best_acc >= 99.0:
                print(f"🎉 %99+ accuracy - MÜKEMMEL!")
                break
            
            if best_acc >= 99.0 and no_improve >= patience:  # %99 hedef
                print(f"✅ Early stop at {best_acc:.1f}% (no improve for {patience} epochs)")
                break
            
            if best_acc < 95.0 and no_improve >= patience * 3:
                print(f"⚠️ Early stop at {best_acc:.1f}% (no improve for {patience * 3} epochs, < 95%)")
                break
            
            if current_epoch >= max_epochs:
                print(f"⚠️ Max epoch ({max_epochs}) - best: {best_acc:.1f}%")
                break
        
        if best_state:
            ft_model.load_state_dict(best_state)
        
        session["status"] = "completed"
        session["accuracy"] = best_acc
        session["epochs"] = current_epoch
        session["epoch"] = current_epoch
        training_progress.update(session); save_session(query_id, session) if "query_id" in dir() and query_id else training_progress
        
        from datetime import datetime; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ft_path = f'../checkpoints/model_finetuned_{timestamp}.pt'
        
        torch.save({
            'model_state_dict': ft_model.state_dict(),
            'model_type': 'v1_finetune',
            'encoder': le,
            'class_names': [str(c) for c in le.classes_],
            'feature_cols': numeric_cols,
            'n_classes': n_classes,
            'input_dim': input_dim,
            'n_sectors': ft_model.n_sectors,
            'target_col': target_col,
            'accuracy': best_acc,
            'config': ft_config
        }, ft_path)
        
        # ONNX export for faster CPU inference
        try:
            onnx_path = ft_path.replace(".pt", ".onnx")
            dummy_input = torch.randn(1, input_dim)
            torch.onnx.export(ft_model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            print(f"ONNX model saved: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
        
        # Temp dosya varsa sil (tek dosya modunda)
        try:
            if 'temp_file' in dir() and temp_file and hasattr(temp_file, 'name'):
                os.unlink(temp_file.name)
        except:
            pass
        
        model_id = f"model_finetuned_{timestamp}"
        session["model_id"] = model_id
        training_progress.update(session); save_session(query_id, session) if "query_id" in dir() and query_id else training_progress
        
        # Sector tahmini yap
        ft_model.eval()
        with torch.inference_mode():
            sample_X = torch.FloatTensor(X[:min(100, len(X))])
            out = ft_model(sample_X)
            sector_probs = torch.softmax(out['sector'], dim=1)
            sector_conf = sector_probs.max(1).values.mean().item() * 100
            dominant_sector = sector_probs.mean(0).argmax().item()
        
        return jsonify({
            "status": "success",
            "accuracy": float(best_acc),
            "epochs": current_epoch,
            "requested_epochs": epochs,
            "n_classes": n_classes,
            "classes": [str(c) for c in le.classes_],
            "model_path": ft_path,
            "model_id": model_id,
            "rows": len(df),
            "sector": {
                "id": dominant_sector,
                "confidence": round(sector_conf, 1),
                "description": f"Data cluster {dominant_sector}"
            },
            "target_column": target_col,
            "miras_enabled": use_miras if 'use_miras' in dir() else False,
            "n_features": input_dim,
            "merged_file_id": merged_file_id if "merged_file_id" in dir() else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/training/progress', methods=['GET'])
def get_training_progress():
    query_id = request.args.get("query_id")
    if query_id and query_id in training_sessions:
        return jsonify(training_sessions[query_id])
    return jsonify(training_progress)

@app.route('/training/progress/<query_id>', methods=['GET'])
def get_training_progress_by_id(query_id):
    if query_id in training_sessions:
        return jsonify(training_sessions[query_id])
    return jsonify({"status": "idle", "query_id": query_id})

@app.route('/files', methods=['GET'])
def list_files():
    try:
        upload_dir = Path('../uploads')
        files = []
        if upload_dir.exists():
            for f in upload_dir.glob('*'):
                if f.is_file():
                    parts = f.name.split('_', 1)
                    file_id = parts[0] if len(parts) > 1 else f.stem
                    filename = parts[1] if len(parts) > 1 else f.name
                    files.append({
                        "file_id": file_id,
                        "filename": filename,
                        "path": str(f),
                        "size": f.stat().st_size
                    })
        return jsonify({"files": files, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =============================================================================
# MIRAS OPTIONS ENDPOINT
# =============================================================================

@app.route('/miras/options', methods=['GET'])
def miras_options():
    """Get available MIRAS configuration options"""
    options = TabularFoundationModelMIRAS.list_miras_options()
    return jsonify({
        "status": "success",
        "miras_options": options,
        "total_features": 49,
        "description": {
            "attentional_bias": "Loss function for memory learning (huber=outlier robust, lp=aggressive)",
            "retention_gate": "How much to forget old memories (lq=stable, kl=probabilistic)",
            "memory_algorithm": "Optimization method for memory updates",
            "architectural": "Additional architectural features",
            "special": "Special capabilities from MIRAS paper"
        },
        "recommended": {
            "default": {"attentional_bias": "huber", "retention_gate": "lq"},
            "noisy_data": {"attentional_bias": "huber", "retention_gate": "elastic"},
            "large_data": {"attentional_bias": "l2", "retention_gate": "l2_local"},
            "small_data": {"attentional_bias": "lp", "retention_gate": "kl"}
        }
    })

@app.route('/miras/info', methods=['GET'])
def miras_info():
    """Get MIRAS framework information"""
    from layers.miras import list_all_features
    features = list_all_features()
    return jsonify({
        "status": "success",
        "framework": "MIRAS (Google Research 2025)",
        "paper": "It's All Connected: A Journey Through Test-Time Memorization",
        "features": features,
        "total": sum(len(v) for v in features.values()),
        "integrated_in": "TabularFoundationModelMIRAS"
    })



# ============= ASYNC TRAINING ENDPOINTS =============
@app.route('/finetune/async', methods=['POST'])
def finetune_async():
    """
    Async fine-tune endpoint - /finetune'u thread'de çalıştırır
    """
    import uuid
    import threading
    import tempfile
    import shutil
    from werkzeug.datastructures import FileStorage
    
    task_id = str(uuid.uuid4())
    
    # Form data'yı kopyala
    form_data = dict(request.form)
    form_data['query_id'] = task_id
    
    # Dosyaları geçici dizine kaydet
    temp_files = []
    if 'file' in request.files:
        files = request.files.getlist('file')
        for f in files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            f.save(temp_file.name)
            temp_file.close()
            temp_files.append(temp_file.name)
    
    # Thread'de finetune çalıştır
    def run_finetune():
        try:
            # Geçici dosyaları aç
            file_storages = []
            for temp_path in temp_files:
                file_storages.append(open(temp_path, 'rb'))
            
            with app.test_request_context(
                method='POST',
                data=form_data,
                content_type='multipart/form-data'
            ):
                # request.files'ı manuel oluştur
                from werkzeug.datastructures import FileStorage, MultiDict
                files_dict = MultiDict()
                for i, f in enumerate(file_storages):
                    files_dict.add('file', FileStorage(f, filename=f'file_{i}.csv'))
                
                with app.request_context((request.environ.copy())):
                    request.files = files_dict
                    finetune()
            
            # Cleanup
            for f in file_storages:
                f.close()
            for temp_path in temp_files:
                try: os.unlink(temp_path)
                except: pass
                
        except Exception as e:
            print(f"Async training error: {e}")
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=run_finetune, daemon=True)
    thread.start()
    
    return jsonify({
        'status': 'started',
        'task_id': task_id,
        'message': 'Training started in background'
    })

@app.route('/training/status/<task_id>', methods=['GET'])
def training_status(task_id):
    """Training status döner (Redis'ten)"""
    from async_training import get_training_status
    status = get_training_status(task_id)
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 6000))
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
