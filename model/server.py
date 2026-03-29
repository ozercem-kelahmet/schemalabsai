import os
import importlib
from analytics_engine import detect_analytics_type, generate_analytics
# smart_analyzer removed - LLM handles column matching directly
os.environ["FLASK_SKIP_DOTENV"] = "1"
import warnings
import tempfile
warnings.filterwarnings("ignore", category=UserWarning)

import storage as cloud_storage

def storage_exists(path):
    gcs_key = path.replace('../', '').replace('./', '')
    basename = os.path.basename(gcs_key)
    # Try exact key
    try:
        if cloud_storage.exists(gcs_key):
            return True
    except:
        pass
    # Try common prefixes in GCS
    if cloud_storage.STORAGE_BACKEND == 'gcs':
        try:
            client = cloud_storage._get_gcs_client()
            bucket = client.bucket(cloud_storage.GCS_BUCKET)
            for prefix in ['shared/checkpoints/', 'shared/base-models/', 'uploads/', 'users/']:
                blobs = list(bucket.list_blobs(prefix=prefix + basename))
                if blobs:
                    return True
                if prefix == 'users/':
                    blobs = list(bucket.list_blobs(prefix=prefix))
                    for b in blobs:
                        if b.name.endswith('/' + basename):
                            return True
        except:
            pass
    return False

def storage_resolve(path):
    gcs_key = path.replace('../', '').replace('./', '')
    basename = os.path.basename(gcs_key)
    # Try exact key
    try:
        return cloud_storage.download(gcs_key)
    except:
        pass
    # Try common prefixes
    if cloud_storage.STORAGE_BACKEND == 'gcs':
        try:
            client = cloud_storage._get_gcs_client()
            bucket = client.bucket(cloud_storage.GCS_BUCKET)
            for prefix in ['shared/checkpoints/', 'shared/base-models/', 'uploads/']:
                blobs = list(bucket.list_blobs(prefix=prefix + basename))
                if blobs:
                    return cloud_storage.download(blobs[0].name)
            # Deep search under users/
            for b in bucket.list_blobs(prefix='users/'):
                if b.name.endswith('/' + basename):
                    return cloud_storage.download(b.name)
        except:
            pass
    return None

def storage_listdir(directory):
    files = []
    try:
        prefix = directory.replace('../', '').replace('./', '')
        if not prefix.endswith('/'):
            prefix += '/'
        if cloud_storage.STORAGE_BACKEND == 'gcs':
            keys = cloud_storage._get_gcs_client().bucket(cloud_storage.GCS_BUCKET).list_blobs(prefix=prefix)
            files = [os.path.basename(b.name) for b in keys if not b.name.endswith('/') and not b.name.endswith('.keep')]
    except:
        pass
    if not files and storage_exists(directory):
        files = storage_listdir(directory)
    return files
from flask import Flask, request, jsonify

def clean_column_name(col):
    """Remove dataset prefixes and make human-readable"""
    import re
    # Remove hash prefixes like "e37c459c_", "a1b2c3d4_"
    col = re.sub(r'^[a-f0-9]{8}_', '', col)
    # Replace underscores with spaces and title case
    col = col.replace('_', ' ').title()
    return col

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
try:
    torch.set_float32_matmul_precision("medium")
except RuntimeError:
    pass
try:
    torch.set_num_threads(8)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd


# GPU accelerated merge with cuDF
try:
    import cudf
    HAS_CUDF = True
except:
    HAS_CUDF = False
from model import TabularFoundationModel, TabularFoundationModelMIRAS
import os
import sys
import time
import threading

# Global GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base model - startup'ta yukle
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "base_model_v0_1M_final.pt")
base_model = None

def load_base_model():
    global base_model
    if base_model is not None:
        return base_model
    
    try:
        from model import TabularFoundationModel
        
        # Load base model from GCS only
        base_path = cloud_storage.download(cloud_storage.shared_key("base-models", os.path.basename(BASE_MODEL_PATH)))
        ckpt = torch.load(base_path, map_location=device, weights_only=False)
        print(f"[BASE MODEL] Loaded from GCS: {base_path}")
        config = ckpt.get("config", {"d_model": 256, "n_heads": 8, "n_layers": 3, "schema_layers": 3, "n_latents": 64, "n_features": 64, "n_classes": 10, "n_sectors": 10, "n_types": 10, "max_cols": 1024})
        base_model = TabularFoundationModel(config).to(device)
        base_model.update_heads(n_classes=18, n_sectors=10)
        if "model_state_dict" in ckpt:
            base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            base_model.load_state_dict(ckpt, strict=False)
        base_model.eval()
        return base_model
    except Exception as e:
        print(f"[BASE MODEL] Error loading: {e}")
        return None

# Load base model at startup
print("[STARTUP] Loading base model...")
load_base_model()
print("[STARTUP] Base model ready")

# Fine-tuned model cache - her seferinde yüklememek için
ft_model_cache = {}
FT_CACHE_MAX_SIZE = 3  # Max 3 model tut (GPU memory için)

def get_cached_finetuned_model(model_id, config, model_path=None):
    """Load model from cache or disk"""
    global ft_model_cache
    
    # Use model_path as cache key if provided
    cache_key = model_path if model_path else model_id
    
    if cache_key in ft_model_cache:
        print(f"Model cache HIT: {cache_key}")
        return ft_model_cache[cache_key]
    
    print(f"Model cache MISS: {cache_key}, loading...")
    
    # Try multiple checkpoint path options
    ft_path = None
    possible_paths = []
    
    # First priority: model_path if provided
    if model_path:
        if model_path.endswith('.pt'):
            possible_paths.append(model_path)
            possible_paths.append(f'../checkpoints/{model_path}')
        else:
            possible_paths.append(f'../checkpoints/{model_path}.pt')
            possible_paths.append(model_path)
    
    # Then try model_id based paths
    possible_paths.extend([
        f'../checkpoints/{model_id}.pt',
        f'../checkpoints/{model_id}',
    ])
    
    # Also try to find by date pattern in model_path or model_id
    import re
    import glob
    search_str = model_path or model_id
    date_match = re.search(r'(\d{8})', search_str)
    if date_match:
        date_str = date_match.group(1)
        pattern = f'../checkpoints/model_finetuned_{date_str}_*.pt'
        matching_files = glob.glob(pattern)
        if matching_files:
            possible_paths.insert(0, matching_files[0])
            print(f"Found checkpoint by date pattern: {matching_files[0]}")
    
    for path in possible_paths:
        resolved = storage_resolve(path)
        if resolved:
            ft_path = resolved
            print(f"[STORAGE] Checkpoint resolved: {path} → {ft_path}")
            break
    
    if not ft_path:
        raise FileNotFoundError(f"No checkpoint found for model: {model_id}, model_path: {model_path}")
    
    print(f"Loading checkpoint from: {ft_path}")
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    ft_ckpt = torch.load(ft_path, map_location=map_location, weights_only=False)
    
    # Use config from checkpoint if available, otherwise use provided config
    ckpt_config = ft_ckpt.get('config', config)
    if ckpt_config:
        config = ckpt_config
    
    ft_model = TabularFoundationModel(config)
    ft_model.load_state_dict(ft_ckpt["model_state_dict"], strict=False)
    ft_model.eval()
    ft_model = ft_model.to(device)
    
    # Cache doluysa en eskiyi sil
    if len(ft_model_cache) >= FT_CACHE_MAX_SIZE:
        oldest_key = list(ft_model_cache.keys())[0]
        print(f"Cache full, removing: {oldest_key}")
        del ft_model_cache[oldest_key]['model']
        del ft_model_cache[oldest_key]
        torch.cuda.empty_cache()
    
    ft_model_cache[cache_key] = {'model': ft_model, 'ckpt': ft_ckpt}
    return ft_model_cache[cache_key]

from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except:
    HAS_STATSMODELS = False
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
def send_training_email(user_email, status, model_name, accuracy=None, error=None, user_name=None):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    sender = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")
    
    if not sender or not password:
        print("Email credentials not configured")
        return
    
    display_name = user_name if user_name else "there"
    
    if status == "completed":
        subject = "Model Training Complete - SchemaLabs AI"
        html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0; padding:0; background:#000000; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#000000; padding:40px 20px;">
<tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" style="background:#0a0a0a; border:1px solid #222; border-radius:12px;">

<tr><td style="padding:35px 40px; border-bottom:1px solid #222;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr>
<td><p style="color:#fff; font-size:20px; font-weight:700; margin:0; letter-spacing:-0.5px;">SchemaLabs<span style="color:#666;">AI</span></p></td>
</tr>
</table>
</td></tr>

<tr><td style="padding:40px;">
<p style="color:#888; font-size:15px; margin:0 0 20px 0;">Hi {display_name},</p>
<h1 style="color:#fff; margin:0 0 12px 0; font-size:28px; font-weight:600;">Training Complete ✓</h1>
<p style="color:#666; font-size:15px; margin:0; line-height:1.6;">Your fine-tuned model is ready to use</p>
</td></tr>

<tr><td style="padding:0 40px 40px 40px;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#111; border:1px solid #222; border-radius:8px;">
<tr>
<td style="padding:24px; border-right:1px solid #222;" width="50%">
<p style="color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 8px 0;">Model</p>
<p style="color:#fff; font-size:16px; margin:0; font-weight:500;">{model_name}</p>
</td>
<td style="padding:24px;" width="50%">
<p style="color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 8px 0;">Accuracy</p>
<p style="color:#fff; font-size:22px; margin:0; font-weight:600;">{accuracy:.1f}%</p>
</td>
</tr>
</table>
</td></tr>

<tr><td style="padding:0 40px 40px 40px;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr><td align="center">
<a href="https://schemalabs.ai" style="display:inline-block; background:#fff; color:#000; padding:14px 40px; border-radius:6px; text-decoration:none; font-weight:600; font-size:14px;">Open Dashboard</a>
</td></tr>
</table>
</td></tr>

<tr><td style="padding:20px 40px; border-top:1px solid #222; background:#050505;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr>
<td><p style="color:#444; font-size:12px; margin:0;">© 2025 SchemaLabs AI</p></td>
<td align="right"><p style="color:#333; font-size:11px; margin:0;">Intelligent Data Analysis</p></td>
</tr>
</table>
</td></tr>

</table>
</td></tr>
</table>
</body>
</html>
"""
    else:
        subject = "Training Failed - SchemaLabs AI"
        html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0; padding:0; background:#000000; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#000000; padding:40px 20px;">
<tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" style="background:#0a0a0a; border:1px solid #222; border-radius:12px;">

<tr><td style="padding:35px 40px; border-bottom:1px solid #222;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr>
<td><p style="color:#fff; font-size:20px; font-weight:700; margin:0; letter-spacing:-0.5px;">SchemaLabs<span style="color:#666;">AI</span></p></td>
</tr>
</table>
</td></tr>

<tr><td style="padding:40px;">
<p style="color:#888; font-size:15px; margin:0 0 20px 0;">Hi {display_name},</p>
<h1 style="color:#fff; margin:0 0 12px 0; font-size:28px; font-weight:600;">Training Failed</h1>
<p style="color:#666; font-size:15px; margin:0; line-height:1.6;">Something went wrong during training</p>
</td></tr>

<tr><td style="padding:0 40px 40px 40px;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#111; border:1px solid #2a2a2a; border-radius:8px;">
<tr><td style="padding:20px;">
<p style="color:#555; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin:0 0 10px 0;">Error Details</p>
<p style="color:#e55; font-size:13px; margin:0; font-family:'SF Mono',Monaco,monospace; word-break:break-all; line-height:1.5;">{error}</p>
</td></tr>
</table>
</td></tr>

<tr><td style="padding:0 40px 15px 40px;">
<p style="color:#555; font-size:13px; margin:0; line-height:1.6;">Please check your data and configuration, then try again.</p>
</td></tr>

<tr><td style="padding:0 40px 40px 40px;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr><td align="center">
<a href="https://schemalabs.ai" style="display:inline-block; background:#fff; color:#000; padding:14px 40px; border-radius:6px; text-decoration:none; font-weight:600; font-size:14px;">Try Again</a>
</td></tr>
</table>
</td></tr>

<tr><td style="padding:20px 40px; border-top:1px solid #222; background:#050505;">
<table width="100%" cellpadding="0" cellspacing="0">
<tr>
<td><p style="color:#444; font-size:12px; margin:0;">© 2025 SchemaLabs AI</p></td>
<td align="right"><p style="color:#333; font-size:11px; margin:0;">Intelligent Data Analysis</p></td>
</tr>
</table>
</td></tr>

</table>
</td></tr>
</table>
</body>
</html>
"""
    
    msg = MIMEMultipart('alternative')
    msg['From'] = f"SchemaLabs AI <{sender}>"
    msg['To'] = user_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {user_email}")
    except Exception as e:
        print(f"Email send failed: {e}")


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
    Smart merge: row concat if same columns, else column merge with prefix
    """
    if len(dataframes) == 0:
        return None
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Check if all dataframes have same columns
    first_cols = set(dataframes[0].columns)
    same_structure = all(set(df.columns) == first_cols for df in dataframes)
    
    if same_structure:
        # Same columns - simple row concat
        merged = pd.concat(dataframes, axis=0, ignore_index=True)
        merged = merged.fillna(0)
        print(f"Row concat (same structure): {len(dataframes)} files -> {merged.shape}")
        return merged
    
    # Different columns - use prefix merge
    print(f"Column merge (different structure): {len(dataframes)} files")
    return smart_merge_with_prefix(dataframes, file_names)

def smart_merge_with_prefix(dataframes, file_names=None):
    """
    Column merge with prefix for different structure files
    """
    global HAS_CUDF
    
    # GPU merge if cuDF available - fully dynamic
    if HAS_CUDF and len(dataframes) > 1:
        try:
            import cudf
            print("Using GPU accelerated merge with cuDF")
            start_time = time.time()
            
            # Find common columns across ALL dataframes (potential merge keys)
            common_cols = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_cols &= set(df.columns)
            
            # Find best merge key dynamically - high cardinality, non-null
            merge_key = None
            best_score = 0
            for col in common_cols:
                try:
                    scores = []
                    for df in dataframes:
                        nunique = df[col].nunique()
                        non_null = df[col].notna().sum() / len(df)
                        scores.append(nunique * non_null)
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        merge_key = col
                except:
                    continue
            
            if merge_key:
                print(f"GPU merge key: {merge_key}")
                # Add prefix to non-merge columns
                prefixed_dfs = []
                for i, df in enumerate(dataframes):
                    prefix = file_names[i].split('_')[0][:8] if file_names and i < len(file_names) else f"f{i}"
                    new_cols = {col: col if col == merge_key else f"{prefix}_{col}" for col in df.columns}
                    prefixed_dfs.append(df.rename(columns=new_cols))
                
                # Convert to cuDF and merge
                cu_dfs = [cudf.DataFrame.from_pandas(df) for df in prefixed_dfs]
                merged = cu_dfs[0]
                for cu_df in cu_dfs[1:]:
                    merged = merged.merge(cu_df, on=merge_key, how='outer')
            else:
                # No common key - simple column concat with prefix
                print("GPU merge: no common key, using column concat")
                prefixed_dfs = []
                for i, df in enumerate(dataframes):
                    prefix = file_names[i].split('_')[0][:8] if file_names and i < len(file_names) else f"f{i}"
                    new_cols = {col: f"{prefix}_{col}" for col in df.columns}
                    prefixed_dfs.append(df.rename(columns=new_cols))
                cu_dfs = [cudf.DataFrame.from_pandas(df) for df in prefixed_dfs]
                merged = cudf.concat(cu_dfs, axis=1)
            
            result = merged.to_pandas()
            print(f"GPU merge completed in {time.time() - start_time:.2f}s, shape: {result.shape}")
            return result.fillna(0)
        except Exception as e:
            print(f"GPU merge failed, falling back to CPU: {e}")
    
    # Original CPU implementation follows

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
        
        # Common columns that should NOT get prefix
        common_cols = ["Player", "Team", "Date", "Name", "player_num", "Rk", "Gcar", "Gtm", "Opp", "Result", "Type", "GS", "MP"]
        
        new_cols = {}
        for col in df.columns:
            if col in common_cols or col.lower() in [c.lower() for c in common_cols]:
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
    
    # Higher LR for small datasets, lower for large
    if n_samples < 500:
        lr = 0.005  # Small dataset - faster learning
    elif n_samples < 2000:
        lr = 0.003  # Medium dataset
    elif batch_size <= 8:
        lr = 0.001
    elif batch_size <= 32:
        lr = 0.002
    else:
        lr = 0.003
    
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



_sector_cache = {}  # model_id -> sector

def detect_sector_with_llm(column_names):
    """Kolon isimlerinden sektor tahmin et"""
    try:
        api_key = os.getenv('SECTOR_MODEL')
        if not api_key:
            return 'unknown'
        
        sector_module = importlib.import_module(os.getenv("SECTOR_CLIENT"))
        client = sector_module.Anthropic(api_key=api_key)
        
        cols_str = ', '.join(column_names[:30])  # İlk 30 kolon
        
        response = client.messages.create(
            model=os.getenv('SECTOR_MODEL_NAME'),
            max_tokens=int(os.getenv('SECTOR_MODEL_MAX_TOKENS')),
            messages=[{
                "role": "user",
                "content": f"Bu veri kolonlarına bakarak sektörü belirle. Sadece tek kelime cevap ver (sports, finance, healthcare, technology, retail, manufacturing, education, entertainment, real_estate, transportation, energy, agriculture, hospitality, government, other): {cols_str}"
            }]
        )
        
        return response.content[0].text.strip().lower()
    except Exception as e:
        print(f"Sector detection error: {e}")
        return 'unknown'


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
    """Load sessions and mark stale trainings as failed"""
    global training_sessions
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'r') as f:
                training_sessions = json.load(f)
                # Mark stale trainings as failed on restart
                for qid, sess in training_sessions.items():
                    if sess.get("status") in ["training", "preparing", "merging", "processing", "starting"]:
                        sess["status"] = "failed"
                        user_id = sess.get("user_id", "")
                        if user_id:
                            try:
                                import psycopg2
                                conn = psycopg2.connect(os.getenv("DATABASE_URL"))
                                cur = conn.cursor()
                                cur.execute("SELECT email, name FROM users WHERE id = %s", (user_id,))
                                result = cur.fetchone()
                                if result:
                                    send_training_email(result[0], "failed", "model", error="Training interrupted by server restart", user_name=result[1] if len(result) > 1 else None)
                                cur.close()
                                conn.close()
                            except Exception as ex:
                                print(f"Email send failed: {ex}")
                        sess["error"] = "Training failed - connection lost"
                        sess["notification"] = "Training failed. Please try again."
                        # Update DB training_failed flag
                        try:
                            import psycopg2
                            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
                            cur = conn.cursor()
                            cur.execute("UPDATE queries SET training_failed = TRUE, is_training = FALSE WHERE id = %s", (qid,))
                            conn.commit()
                            cur.close()
                            conn.close()
                        except Exception as db_ex:
                            print(f"DB update failed: {db_ex}")
    except:
        training_sessions = {}

# Global Kafka producer
_kafka_producer = None
def _get_kafka_producer():
    global _kafka_producer
    import os as _os2
    kafka_servers = _os2.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if not kafka_servers:
        return None
    try:
        if _kafka_producer is None:
            from kafka import KafkaProducer
            _kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000,
                max_block_ms=5000
            )
            print("[KAFKA] Producer initialized")
        return _kafka_producer
    except Exception as e:
        print(f"[KAFKA INIT ERROR] {e}")
        return None

# Global Kafka producer
_kafka_producer = None
def _get_kafka_producer():
    global _kafka_producer
    import os as _os2
    kafka_servers = _os2.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if not kafka_servers:
        return None
    try:
        if _kafka_producer is None:
            from kafka import KafkaProducer
            _kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000,
                max_block_ms=5000
            )
            print("[KAFKA] Producer initialized")
        return _kafka_producer
    except Exception as e:
        print(f"[KAFKA INIT ERROR] {e}")
        return None

def _save_sessions():
    try:
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(training_sessions, f)
    except Exception as e:
        print(f"[SESSION] Failed to save sessions: {e}")

def get_session(query_id):
    _load_sessions()
    if query_id not in training_sessions:
        training_sessions[query_id] = {"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "idle", "eta": "0%", "start_time": 0, "query_id": query_id}
        _save_sessions()
    return training_sessions[query_id]

def _get_redis():
    try:
        import redis as _redis
        url = os.getenv("REDIS_URL", "localhost:6379")
        host, port = url.split(":")
        return _redis.Redis(host=host, port=int(port), password=os.getenv("REDIS_PASSWORD", ""), decode_responses=True)
    except:
        return None

_session_lock = __import__('threading').Lock()

def save_session(query_id, session):
    with _session_lock:
        training_sessions[query_id] = session
        if "history" not in session:
            session["history"] = []
        ep = session.get("epoch", 0)
        ac = session.get("accuracy", 0)
        lo = session.get("loss", 0)
        if ep > 0 and (len(session["history"]) == 0 or session["history"][-1].get("epoch", 0) < ep):
            session["history"].append({"epoch": ep, "accuracy": ac, "loss": lo})
            print(f"[REDIS-HISTORY] qid={query_id} epoch={ep} history_len={len(session['history'])}")
    try:
        rc = _get_redis()
        if rc:
            rc.setex(f"training:{query_id}", 86400, json.dumps(session, default=str))
    except Exception as e:
        print(f"[SESSION] Redis write failed: {e}")
    
      # Kafka event - global producer ile
    try:
        if ep > 0:
            _kp = _get_kafka_producer()
            if _kp:
                _kp.send("training_progress", {
                    "query_id": query_id,
                    "epoch": ep,
                    "accuracy": ac,
                    "loss": lo,
                    "status": session.get("status", "training"),
                    "epochs": session.get("epochs", 0),
                    "user_id": session.get("user_id", ""),
                    "model_id": session.get("model_id", ""),
                    "learningRate": session.get("lr", 0)
                })
                _kp.flush(timeout=2)
                print(f"[KAFKA] Sent: qid={query_id} epoch={ep}")
    except Exception as _ke:
        print(f"[KAFKA ERROR] {_ke}")

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
        
        # Load checkpoint to get source_file_id
        source_file = None
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
            source_file = ckpt.get('source_file_id')
        except:
            pass
        
        models.append({
            "name": filename.replace('.pt', ''),
            "filename": filename,
            "path": str(f),
            "type": "finetuned" if "finetuned" in filename else "base",
            "is_current": filename == "schemalabsai_v1.pt",
            "source_file_id": source_file
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
            
            X_t = torch.FloatTensor(row_norm).to(device)
            mask_t = torch.FloatTensor(mask.astype(np.float32)).to(device)
            
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
        print("="*80)
        print("❌ TRAINING EXCEPTION")
        print("="*80)
        import traceback
        traceback.print_exc()
        print(f"best_acc at exception: {best_acc if 'best_acc' in locals() else 'UNDEFINED'}")
        print("="*80)
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_csv():
    """Batch prediction for CSV file upload (Spark preprocessed)"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        model_id = request.form.get('model_id', None)
        
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_file.name)
        temp_file.close()
        
        df = pd.read_csv(temp_file.name)
        os.unlink(temp_file.name)
        
        print(f"[BATCH PREDICT] {len(df)} rows, {len(df.columns)} cols, model={model_id}")
        
        # Load model
        ckpt = None
        if model_id:
            ckpt_path = f"./finetuned_models/{model_id}.pt"
            if storage_exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
        
        if ckpt is None:
            return jsonify({"error": f"Model {model_id} not found"}), 404
        
        ft_model = load_finetuned_model(ckpt)
        if ft_model is None:
            return jsonify({"error": "Failed to load model"}), 500
        
        # Preprocess
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) == 0:
            return jsonify({"error": "No numeric features"}), 400
        
        X = numeric_df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        
        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            X = ckpt['scaler'].transform(X)
        
        # Batch predict - 1000 satır chunk
        all_preds = []
        all_confs = []
        chunk_size = 1000
        
        ft_model.eval()
        with torch.inference_mode():
            for i in range(0, len(X), chunk_size):
                chunk = torch.FloatTensor(X[i:i+chunk_size]).to(device)
                out = ft_model(chunk)
                if isinstance(out, dict):
                    logits = out.get('logits', out.get('output', list(out.values())[0]))
                else:
                    logits = out
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(1)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_confs.extend(conf.cpu().numpy().tolist())
        
        # Le classes
        le = ckpt.get('label_encoder', None)
        class_names = le.classes_.tolist() if le else [str(i) for i in range(len(set(all_preds)))]
        labels = [class_names[p] if p < len(class_names) else str(p) for p in all_preds]
        
        df['prediction'] = labels
        df['confidence'] = all_confs
        
        print(f"[BATCH PREDICT] Done: {len(df)} predictions")
        
        return jsonify({
            "status": "ok",
            "rows": len(df),
            "predictions": labels,
            "confidences": all_confs,
            "columns": df.columns.tolist()
        })
        
    except Exception as e:
        import traceback
        print(f"[BATCH PREDICT ERROR] {e}")
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
        
        X_t = torch.FloatTensor(values_norm).to(device)
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
        print("="*80)
        print("❌ TRAINING EXCEPTION")
        print("="*80)
        import traceback
        traceback.print_exc()
        print(f"best_acc at exception: {best_acc if 'best_acc' in locals() else 'UNDEFINED'}")
        print("="*80)
        return jsonify({"error": str(e)}), 500


# ============ DATA CACHE ============
data_cache = {}
DATA_CACHE_MAX_SIZE = 100

def get_cached_dataframe(file_path):
    global data_cache
    if file_path in data_cache:
        print(f"Data cache HIT: {os.path.basename(file_path)}")
        return data_cache[file_path].copy()
    print(f"Data cache MISS: {os.path.basename(file_path)}")
    resolved = storage_resolve(file_path)
    if not resolved:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(resolved)
    if len(data_cache) >= DATA_CACHE_MAX_SIZE:
        del data_cache[list(data_cache.keys())[0]]
    data_cache[file_path] = df
    return df.copy()

aggregate_cache = {}


# Query result cache - LLM responses
query_cache = {}
QUERY_CACHE_MAX_SIZE = 100

def get_cached_query_result(file_id, query):
    """Get cached LLM result for (file_id, query) pair"""
    cache_key = f"{file_id}_{query.lower().strip()[:100]}"  # First 100 chars of query
    if cache_key in query_cache:
        print(f"Query cache HIT: {query[:50]}...")
        return query_cache[cache_key]
    return None

def set_cached_query_result(file_id, query, result):
    """Cache LLM result"""
    global query_cache
    cache_key = f"{file_id}_{query.lower().strip()[:100]}"
    
    if len(query_cache) >= QUERY_CACHE_MAX_SIZE:
        # Remove oldest
        oldest_key = list(query_cache.keys())[0]
        del query_cache[oldest_key]
    
    query_cache[cache_key] = result
    print(f"Query cached: {query[:50]}...")


# Pre-computed statistics cache
stats_cache = {}

def get_cached_statistics(file_path, df):
    """Get or compute basic statistics for all numeric columns"""
    if file_path in stats_cache:
        print(f"Statistics cache HIT: {os.path.basename(file_path)}")
        return stats_cache[file_path]
    
    print(f"Statistics cache MISS: Computing statistics...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {
        'min': {},
        'max': {},
        'mean': {},
        'std': {},
        'median': {},
        'count': {}
    }
    
    for col in num_cols:
        try:
            stats['min'][col] = float(df[col].min())
            stats['max'][col] = float(df[col].max())
            stats['mean'][col] = float(df[col].mean())
            stats['std'][col] = float(df[col].std())
            stats['median'][col] = float(df[col].median())
            stats['count'][col] = int(df[col].count())
        except:
            pass
    
    stats_cache[file_path] = stats
    return stats

def get_cached_aggregate(file_path, group_col):
    key = f"{file_path}_{group_col}"
    return aggregate_cache.get(key)

def set_cached_aggregate(file_path, group_col, result):
    key = f"{file_path}_{group_col}"
    aggregate_cache[key] = result


def generate_multidim_insights(df, num_cols, stats):
    """Generate correlation, outliers, and trend insights"""
    insights = ""
    
    # 1. CORRELATION MATRIX
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols[:50]].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr:
                insights += "\n=== STRONG CORRELATIONS (|r| > 0.7) ===\n"
                high_corr = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]
                for col1, col2, corr in high_corr:
                    insights += f"{col1} ↔ {col2}: {corr:.3f}\n"
        except:
            pass
    
    # 2. ML-BASED ANOMALY DETECTION (IsolationForest)
    ml_anomalies = []
    try:
        # Use top numeric columns for anomaly detection
        anomaly_cols = [c for c in num_cols[:20] if df[c].notna().sum() > 10]
        if len(anomaly_cols) >= 2 and len(df) >= 10:
            X = df[anomaly_cols].dropna()
            if len(X) >= 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
                predictions = iso_forest.fit_predict(X)
                anomaly_indices = X.index[predictions == -1]
                
                if len(anomaly_indices) > 0:
                    insights += f"\n=== ML ANOMALY DETECTION ({len(anomaly_indices)} anomalies found) ===\n"
                    insights += f"Using columns: {', '.join(anomaly_cols[:5])}\n"
                    
                    # Show which columns have highest anomaly scores
                    for col in anomaly_cols[:5]:
                        anomaly_vals = df.loc[anomaly_indices, col].dropna()
                        if len(anomaly_vals) > 0:
                            insights += f"{col}: {len(anomaly_vals)} anomalous values\n"
    except Exception as e:
        print(f"ML Anomaly detection error: {e}")
    
    # 2b. STATISTICAL OUTLIERS (IQR - backup method)
    outlier_cols = []
    for col in num_cols[:30]:
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outlier_cols.append((col, len(outliers), len(df)))
        except:
            pass
    
    if outlier_cols:
        insights += "\n=== STATISTICAL OUTLIERS (IQR) ===\n"
        outlier_cols = sorted(outlier_cols, key=lambda x: -x[1])[:10]
        for col, count, total in outlier_cols:
            pct = (count/total)*100
            insights += f"{col}: {count} outliers ({pct:.1f}%)\n"
    
    # 3. VARIANCE ANALYSIS
    variance_data = []
    for col in num_cols[:50]:
        try:
            if col in stats['std'] and col in stats['mean']:
                cv = stats['std'][col] / stats['mean'][col] if stats['mean'][col] != 0 else 0
                if cv > 0.5:
                    variance_data.append((col, cv))
        except:
            pass
    
    if variance_data:
        insights += "\n=== HIGH VARIABILITY METRICS (CV > 0.5) ===\n"
        variance_data = sorted(variance_data, key=lambda x: -x[1])[:10]
        for col, cv in variance_data:
            insights += f"{col}: CV={cv:.2f}\n"
    
    # 4. PREDICTIVE INSIGHTS (Regression-based)
    try:
        if len(num_cols) >= 2 and len(df) >= 20:
            # Find top correlated pairs for prediction
            corr_matrix = df[num_cols[:30]].corr()
            predictions = []
            
            for i in range(min(5, len(corr_matrix.columns))):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.6:  # Strong correlation
                        col_x = corr_matrix.columns[i]
                        col_y = corr_matrix.columns[j]
                        
                        # Fit simple linear regression
                        X = df[[col_x]].dropna()
                        y = df.loc[X.index, col_y]
                        valid = ~y.isna()
                        X = X[valid]
                        y = y[valid]
                        
                        if len(X) >= 10:
                            model = LinearRegression()
                            model.fit(X, y)
                            slope = model.coef_[0]
                            
                            # Generate prediction insight
                            if slope > 0:
                                predictions.append((col_x, col_y, slope, "increases"))
                            else:
                                predictions.append((col_x, col_y, abs(slope), "decreases"))
            
            if predictions:
                insights += "\n=== PREDICTIVE INSIGHTS ===\n"
                predictions = sorted(predictions, key=lambda x: -abs(x[2]))[:5]
                for x_col, y_col, slope, direction in predictions:
                    insights += f"When {x_col} increases by 1, {y_col} {direction} by {slope:.2f}\n"
    except Exception as e:
        print(f"Predictive insights error: {e}")
    
    # 5. TIME SERIES TREND ANALYSIS
    try:
        if HAS_STATSMODELS:
            # Detect time columns
            time_cols = [col for col in df.columns if any(t in col.lower() for t in ['date', 'time', 'timestamp'])]
            
            if time_cols and len(num_cols) > 0:
                time_col = time_cols[0]
                
                # Try to parse as datetime
                try:
                    df_time = df[[time_col] + num_cols[:5]].copy()
                    df_time[time_col] = pd.to_datetime(df_time[time_col])
                    df_time = df_time.sort_values(time_col)
                    df_time = df_time.set_index(time_col)
                    
                    # Analyze top numeric column
                    for val_col in num_cols[:3]:
                        ts_data = df_time[val_col].dropna()
                        
                        if len(ts_data) >= 14:  # Need at least 2 periods
                            # Detect trend
                            decomposition = seasonal_decompose(ts_data, model='additive', period=min(7, len(ts_data)//2), extrapolate_trend='freq')
                            
                            trend = decomposition.trend.dropna()
                            if len(trend) >= 2:
                                trend_direction = "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing"
                                trend_change = abs(trend.iloc[-1] - trend.iloc[0])
                                
                                if not insights.endswith("\n"):
                                    insights += "\n"
                                insights += "\n=== TIME SERIES TRENDS ===\n"
                                insights += f"{val_col}: {trend_direction} trend (change: {trend_change:.2f})\n"
                                
                                # Check seasonality strength
                                seasonal_strength = decomposition.seasonal.std() / ts_data.std() if ts_data.std() > 0 else 0
                                if seasonal_strength > 0.3:
                                    insights += f"  → Strong seasonal pattern detected (strength: {seasonal_strength:.2f})\n"
                                
                                break  # Only show first column with trend
                except Exception as e:
                    print(f"Time series decomposition error: {e}")
    except Exception as e:
        print(f"Time series analysis error: {e}")
    
    return insights


@app.route('/analyze', methods=['POST'])
def analyze():
    """Smart analyzer - query-aware, token-efficient"""
    try:
        data = request.json
        file_id = data.get('file_id', '')
        model_id = data.get('model_id', '')  # Fine-tuned model ID or "schema-v0" for base
        query = data.get('query', data.get('message', '')).lower()
        user_id = data.get('user_id', '')
        print(f'[VERTICAL DEBUG] model_id={model_id}, user_id={user_id}')
        use_base_model = (model_id == "schema-v0" or model_id == "" or model_id == "none")
        
        uploads_dir = '../uploads'
        file_path = None
        
        # If using base model (schema-v0), skip fine-tuned model loading
        if use_base_model:
            print(f"[ANALYZE] Using base model (schema-v0)")
            # Base model analizi - sadece veri analizi yap, model inference yok
            pass
        # If model_id exists, try to get source_file_id (merged file) from checkpoint
        elif model_id:
            try:
                # Try model_path first (from database), then fallback to model_id.pt
                model_path = data.get('model_path', '')
                print(f"DEBUG: Received model_path: {model_path}")
                if model_path and storage_exists(model_path):
                    ft_path = model_path
                elif model_path:
                    # Try relative path
                    ft_path = model_path if model_path.startswith('../') else f'../checkpoints/{model_path}'
                else:
                    ft_path = f'../checkpoints/{model_id}.pt'
                
                # Also try model_id.pt as fallback
                if not storage_exists(ft_path):
                    ft_path = f'../checkpoints/{model_id}.pt'
                
                print(f"DEBUG: Loading checkpoint from: {ft_path}")
                resolved_ckpt = storage_resolve(ft_path)
                if resolved_ckpt:
                    ft_ckpt = torch.load(resolved_ckpt, map_location='cpu', weights_only=False)
                    source_file_id = ft_ckpt.get('source_file_id', '')
                    if source_file_id and 'merged' in source_file_id:
                        # Use merged file directly
                        merged_path = os.path.join(uploads_dir, source_file_id)
                        if storage_exists(merged_path):
                            file_path = merged_path
                            print(f"Using merged file from model: {file_path}")
                        else:
                            # Try to find by prefix
                            for f in storage_listdir(uploads_dir):
                                if source_file_id[:8] in f and 'merged' in f:
                                    file_path = os.path.join(uploads_dir, f)
                                    print(f"Found merged file: {file_path}")
                                    break
            except Exception as e:
                print(f"Could not load model for source_file_id: {e}")
        
        # === ROBUST FILE SEARCH (6 methods) ===
        
        # Method 1: DB source_name from fine_tuned_models
        if not file_path and model_id and not use_base_model:
            try:
                import psycopg2
                db_url = os.environ.get('DATABASE_URL', 'postgresql://schemalabs:schemalabs@localhost:5432/schemalabs')
                conn = psycopg2.connect(db_url)
                cur = conn.cursor()
                # Get source_name from fine_tuned_models
                cur.execute("SELECT source_name, source_file_id FROM fine_tuned_models WHERE id=%s", (model_id,))
                row = cur.fetchone()
                if row:
                    src_name, src_file_id = row[0], row[1]
                    # Try direct source_name match
                    if src_name:
                        direct_path = os.path.join(uploads_dir, src_name)
                        if storage_exists(direct_path):
                            file_path = direct_path
                            print(f"[FILE SEARCH] Method 1a: DB source_name direct: {file_path}")
                        else:
                            for fname in storage_listdir(uploads_dir):
                                if src_name in fname:
                                    file_path = os.path.join(uploads_dir, fname)
                                    print(f"[FILE SEARCH] Method 1b: DB source_name partial: {file_path}")
                                    break
                    # Try uploaded_files table with source_file_id (may be comma-separated list)
                    if not file_path and src_file_id:
                        file_ids = [x.strip() for x in src_file_id.split(",") if x.strip()]
                        if len(file_ids) > 1:
                            # Multiple files - load all and merge
                            multi_dfs = []
                            for fid in file_ids:
                                cur.execute("SELECT filename, path FROM uploaded_files WHERE id=%s", (fid,))
                                frow = cur.fetchone()
                                if frow:
                                    uf_filename, uf_path = frow[0], frow[1]
                                    for try_path in [
                                        os.path.join(uploads_dir, uf_filename) if uf_filename else None,
                                        os.path.join('..', uf_path) if uf_path else None,
                                    ]:
                                        if try_path and storage_exists(try_path):
                                            try:
                                                resolved_try = storage_resolve(try_path) or try_path
                                                df = pd.read_csv(resolved_try, low_memory=False)
                                                multi_dfs.append(df)
                                            except: pass
                                            break
                            if multi_dfs:
                                import tempfile
                                merged_df = pd.concat(multi_dfs, axis=0, ignore_index=True).fillna(0)
                                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir=uploads_dir)
                                merged_df.to_csv(tmp.name, index=False)
                                file_path = tmp.name
                                print(f"[FILE SEARCH] Method 2-multi: merged {len(multi_dfs)} files -> {merged_df.shape}")
                        else:
                            cur.execute("SELECT filename, path FROM uploaded_files WHERE id=%s", (src_file_id,))
                            frow = cur.fetchone()
                            if frow:
                                uf_filename, uf_path = frow[0], frow[1]
                                for try_path in [
                                    os.path.join('..', uf_path) if uf_path else None,
                                    os.path.join(uploads_dir, uf_filename) if uf_filename else None,
                                ]:
                                    if try_path and storage_exists(try_path):
                                        file_path = try_path
                                        print(f"[FILE SEARCH] Method 2: uploaded_files table: {file_path}")
                                        break
                                if not file_path and uf_filename:
                                    for fname in storage_listdir(uploads_dir):
                                        if uf_filename in fname or fname in uf_filename:
                                            file_path = os.path.join(uploads_dir, fname)
                                            print(f"[FILE SEARCH] Method 2b: uploaded_files partial: {file_path}")
                                            break
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[FILE SEARCH] DB lookup failed: {e}")

        # Method 3: file_id prefix match in uploads dir
        if not file_path and storage_exists(uploads_dir):
            matching_files = []
            for f in storage_listdir(uploads_dir):
                # Exact match with full file_id
                if file_id and (f.startswith(file_id + "_") or f.startswith(file_id + ".")):
                    full_path = os.path.join(uploads_dir, f)
                    matching_files.append((full_path, os.path.getmtime(full_path), 'exact'))
                # Prefix match with first 8 chars
                elif file_id and len(file_id) >= 8 and f.startswith(file_id[:8]):
                    full_path = os.path.join(uploads_dir, f)
                    matching_files.append((full_path, os.path.getmtime(full_path), 'prefix'))
                # Method 4: file_id anywhere in filename
                elif file_id and file_id in f:
                    full_path = os.path.join(uploads_dir, f)
                    matching_files.append((full_path, os.path.getmtime(full_path), 'contains'))
            
            if matching_files:
                exact = [m for m in matching_files if m[2] == 'exact']
                if exact:
                    exact.sort(key=lambda x: x[1], reverse=True)
                    file_path = exact[0][0]
                else:
                    matching_files.sort(key=lambda x: x[1], reverse=True)
                    file_path = matching_files[0][0]
                print(f"Selected file: {file_path}")
        
        if not file_path:
            return jsonify({'analysis': 'File not found.', 'status': 'error'})
        
        resolved_path = storage_resolve(file_path) or file_path
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(resolved_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(resolved_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(resolved_path)
        else:
            df = get_cached_dataframe(file_path)
        
        # === FINE-TUNED MODEL PREDICTION ===
        ft_prediction_text = ""
        ft_structured = None
        if model_id and model_id != "none":
            try:
                # Try model_path first (from database), then fallback to model_id.pt
                model_path = data.get('model_path', '')
                print(f"DEBUG: Received model_path: {model_path}")
                if model_path and storage_exists(model_path):
                    ft_path = model_path
                elif model_path:
                    # Try relative path
                    ft_path = model_path if model_path.startswith('../') else f'../checkpoints/{model_path}'
                else:
                    ft_path = f'../checkpoints/{model_id}.pt'
                
                # Also try model_id.pt as fallback
                if not storage_exists(ft_path):
                    ft_path = f'../checkpoints/{model_id}.pt'
                
                print(f"DEBUG: Loading checkpoint from: {ft_path}")
                resolved_ckpt = storage_resolve(ft_path)
                if resolved_ckpt:
                    ft_ckpt = torch.load(resolved_ckpt, map_location='cpu', weights_only=False)
                    
                    # Get model info
                    class_names = ft_ckpt.get('class_names', [])
                    feature_cols = ft_ckpt.get('feature_cols', [])
                    config = ft_ckpt.get('config', {})
                    n_classes = ft_ckpt.get('n_classes', len(class_names))
                    input_dim = ft_ckpt.get('input_dim', config.get('n_features', 110))
                    accuracy = ft_ckpt.get('accuracy', 0)
                    
                    # Rebuild config if needed
                    if not config:
                        config = {
                            'd_model': 128, 'n_heads': 4, 'n_layers': 2,
                            'schema_layers': 2, 'n_latents': 16,
                            'n_features': input_dim, 'n_classes': n_classes,
                            'vocab_size': 50000, 'n_types': 10, 'max_cols': 128
                        }
                    else:
                        config['n_features'] = input_dim
                        config['n_classes'] = n_classes
                    
                    # Get model from cache or load
                    cached = get_cached_finetuned_model(model_id, config, data.get("model_path", ""))
                    ft_model = cached['model']
                    ft_ckpt = cached['ckpt']  # Update with cached checkpoint
                    
                    # Get input_dim from cached checkpoint (more reliable)
                    input_dim = ft_ckpt.get('input_dim', ft_ckpt.get('config', {}).get('n_features', input_dim))
                    print(f"DEBUG: Using input_dim={input_dim} from checkpoint")
                    
                    # Prepare data for prediction - sample for large datasets
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    df_sample = df.head(10000) if len(df) > 10000 else df
                    X_pred = df_sample[num_cols].fillna(0).values.astype(np.float32)
                    print(f"Fine-tuned prediction: using {len(df_sample)}/{len(df)} rows, X_pred.shape={X_pred.shape}, input_dim={input_dim}")
                    
                    # Match feature dimensions to checkpoint's expected input
                    if X_pred.shape[1] < input_dim:
                        X_pred = np.hstack([X_pred, np.zeros((X_pred.shape[0], input_dim - X_pred.shape[1]))])
                    elif X_pred.shape[1] > input_dim:
                        X_pred = X_pred[:, :input_dim]
                    
                    # Normalize using saved scaler if available
                    if 'scaler' in ft_ckpt and ft_ckpt['scaler'] is not None:
                        try:
                            X_pred = ft_ckpt['scaler'].transform(X_pred)
                        except:
                            X_pred = (X_pred - X_pred.mean(axis=0)) / (X_pred.std(axis=0) + 1e-8)
                    else:
                        X_pred = (X_pred - X_pred.mean(axis=0)) / (X_pred.std(axis=0) + 1e-8)
                    
                    X_pred = np.nan_to_num(X_pred, nan=0.0).astype(np.float32)
                    
                    # Run prediction in batches to avoid OOM
                    batch_size = 1000  # Process 1000 rows at a time
                    all_preds = []
                    all_confs = []
                    
                    with torch.no_grad():
                        for i in range(0, len(X_pred), batch_size):
                            batch = torch.FloatTensor(X_pred[i:i+batch_size]).to(device)
                            output = ft_model(batch)
                            
                            if isinstance(output, dict):
                                logits = output.get('sector', output.get('logits', None))
                                if logits is None:
                                    logits = list(output.values())[0]
                            else:
                                logits = output
                            
                            probs = torch.softmax(logits, dim=-1)
                            all_preds.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
                            all_confs.extend(probs.max(dim=-1).values.cpu().numpy().tolist())
                    
                    preds = np.array(all_preds)
                    confs = np.array(all_confs)
                    
                    # Model is cached, no cleanup needed
                    # torch.cuda.empty_cache() called only when cache is full
                    
                    # Skip the old single-batch code
                    output = None
                        
                    # Batch processing already done above
                    if output is not None:
                        if isinstance(output, dict):
                            logits = output.get('sector', output.get('logits', None))
                            if logits is None:
                                logits = list(output.values())[0]
                        else:
                            logits = output
                        
                        probs = torch.softmax(logits, dim=-1)
                        preds = probs.argmax(dim=-1).numpy()
                        confs = probs.max(dim=-1).values.numpy()
                    
                    # Build structured prediction data
                    from collections import Counter
                    pred_counts = Counter(preds)
                    
                    # Structured predictions list
                    structured_predictions = []
                    for i in range(min(100, len(preds))):  # First 100 rows
                        cls_name = class_names[preds[i]] if preds[i] < len(class_names) else f"class_{preds[i]}"
                        structured_predictions.append({
                            "row": i + 1,
                            "label": cls_name,
                            "confidence": round(float(confs[i]), 4)
                        })
                    
                    # Class distribution
                    class_distribution = {}
                    for cls_idx, count in pred_counts.most_common():
                        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                        class_distribution[cls_name] = {
                            "count": int(count),
                            "percentage": round(count / len(preds) * 100, 2)
                        }
                    
                    # Store structured data for response
                    ft_structured = {
                        "model_id": model_id,
                        "training_accuracy": round(float(accuracy)/100, 4) if accuracy > 1 else round(float(accuracy), 4),
                        "classes": class_names[:20],
                        "total_predictions": len(preds),
                        "predictions": structured_predictions,
                        "class_distribution": class_distribution
                    }
                    
                    # Also keep text version for backward compatibility
                    ft_prediction_text = f"\n=== FINE-TUNED MODEL PREDICTIONS ===\n"
                    ft_prediction_text += f"Model: {model_id}\n"
                    ft_prediction_text += f"Training Accuracy: {accuracy:.1f}%\n" if accuracy > 1 else f"Training Accuracy: {accuracy*100:.1f}%\n"
                    ft_prediction_text += f"Classes: {', '.join(class_names[:10])}{'...' if len(class_names) > 10 else ''}\n"
                    ft_prediction_text += f"Total Predictions: {len(preds)}\n"
                    
                    print(f"Fine-tuned model prediction completed: {len(preds)} rows, {n_classes} classes")
                else:
                    ft_prediction_text = f"\n[Fine-tuned model not found: {model_id}]\n"
            except Exception as e:
                ft_prediction_text = f"\n[Fine-tuned model error: {str(e)}]\n"
                print(f"Fine-tuned prediction error: {e}")
                import traceback
                traceback.print_exc()
        
        # === ANALYTICS ENGINE (Parallel with model prediction) ===
        analytics_result = {'detected': None, 'analysis': None}
        
        def run_analytics():
            detected_types = detect_analytics_type(query)
            if detected_types and detected_types[0]['score'] >= 8:
                print(f"Analytics type detected: {detected_types[0]['type']} (score: {detected_types[0]['score']})")
                advanced_analysis = generate_analytics(df, query, detected_types)
                analytics_result['detected'] = detected_types
                analytics_result['analysis'] = advanced_analysis
        
        # Start analytics in parallel (already computed ft_prediction above)
        analytics_thread = threading.Thread(target=run_analytics)
        analytics_thread.start()
        
        # Wait for analytics to complete
        analytics_thread.join(timeout=10)  # Max 10 sec
        
        # If analytics found something, use it
        if analytics_result['analysis'] and len(analytics_result['analysis']) > 100:
            return jsonify({'analysis': ft_prediction_text + analytics_result['analysis'], 'status': 'success'})
        
        # Normal queries: Model prediction + data context -> LLM handles the rest
        
        # === COMPACT ANALYSIS (max 8K chars) ===
        # Filter out ID columns from numeric columns
        all_num_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_cols = []
        id_cols = []
        for col in all_num_cols:
            col_lower = col.lower()
            # Check if column name suggests ID
            is_id_name = any(x in col_lower for x in ['_id', 'id_', '.id', 'index', '_key', 'key_'])
            is_id_name = is_id_name or col_lower.endswith('id') or col_lower == 'id'
            # Check if values suggest ID pattern (high cardinality, sequential)
            is_id_values = is_id_pattern(df[col], len(df)) if len(df) > 0 else False
            
            if is_id_name or is_id_values:
                id_cols.append(col)
            else:
                num_cols.append(col)
        
        if id_cols:
            print(f"Detected ID columns (excluded from stats): {id_cols[:5]}")
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # === PROFESSIONAL DATA CONTEXT (LLM handles column matching) ===
        analysis = ft_prediction_text
        analysis += f"\n=== DATA OVERVIEW ===\n"
        analysis += f"Total: {len(df)} rows, {len(df.columns)} columns\n"
        analysis += f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}\n\n"
        
        # List all numeric columns with basic stats (ID columns already filtered)
        analysis += "=== NUMERIC COLUMNS ===\n"
        for col in num_cols[:50]:  # First 50
            try:
                analysis += f"{col}: avg={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}\n"
            except:
                pass
        if len(num_cols) > 50:
            analysis += f"... and {len(num_cols) - 50} more numeric columns\n"
        
        # List categorical columns with unique counts
        analysis += "\n=== CATEGORICAL COLUMNS ===\n"
        for col in cat_cols[:20]:  # First 20
            try:
                nunique = df[col].nunique()
                sample_vals = df[col].dropna().unique()[:5]
                analysis += f"{col}: {nunique} unique values (e.g., {', '.join(str(v) for v in sample_vals)})\n"
            except:
                pass
        if len(cat_cols) > 20:
            analysis += f"... and {len(cat_cols) - 20} more categorical columns\n"
        
        
        # === MULTI-DIMENSIONAL INSIGHTS ===
        try:
            stats = get_cached_statistics(file_path, df)
            insights = generate_multidim_insights(df, num_cols, stats)
            if insights:
                analysis += insights + "\n"
        except Exception as e:
            print(f"Insights generation error: {e}")
        
        analysis += "\n"
        # === AGGREGATED DATA (dynamic grouping) ===
        group_col = None
        
        # Priority 1: Columns with "name" - VALIDATE content!
        for col in cat_cols:
            if 'name' in col.lower() and 'file' not in col.lower() and 'user' not in col.lower():
                nunique = df[col].nunique()
                if 2 <= nunique <= 100:
                    # Validate: not "[Word] [Number]" pattern
                    sample_vals = df[col].dropna().head(20).astype(str).tolist()
                    if len(sample_vals) > 0:
                        pattern_count = 0
                        for v in sample_vals:
                            v_lower = v.lower().strip()
                            parts = v_lower.split()
                            if len(parts) >= 2 and parts[-1].isdigit():
                                pattern_count += 1
                        if pattern_count / len(sample_vals) < 0.5:
                            group_col = col
                            print(f"SERVER AGGREGATION: Using {col} (validated)")
                            break
                        else:
                            print(f"SERVER: Skipping {col} - contains codes")
                            continue
        
        # Priority 2: Exclude technical patterns AND Unnamed columns
        if not group_col:
            technical = ['_id', 'id_', '_num', 'num_', '_code', 'code_', '_key', '_uuid', '_index', 'unnamed']
            for col in cat_cols:
                if any(t in col.lower() for t in technical):
                    continue
                nunique = df[col].nunique()
                if 2 <= nunique <= 100:
                    # VALIDATE CONTENT
                    sample_vals = df[col].dropna().head(20).astype(str).tolist()
                    if len(sample_vals) > 0:
                        pattern_count = 0
                        for v in sample_vals:
                            v_lower = v.lower().strip()
                            parts = v_lower.split()
                            if len(parts) >= 2 and parts[-1].isdigit():
                                pattern_count += 1
                        if pattern_count / len(sample_vals) < 0.5:
                            group_col = col
                            print(f"SERVER: Using {col} (validated)")
                            break
        
        # Priority 3: Any categorical (last resort)
        if not group_col:
            for col in cat_cols:
                nunique = df[col].nunique()
                if 2 <= nunique <= 100:
                    group_col = col
                    break
        
        if group_col and num_cols:
            # Check aggregate cache first
            cached_agg = get_cached_aggregate(file_path, group_col)
            if cached_agg is not None:
                analysis += cached_agg
            else:
                print(f"AGGREGATION: Using group_col={group_col}, nunique={df[group_col].nunique()}")
                agg_start = f"=== AGGREGATED BY {group_col} ===\n"
                analysis += agg_start
                try:
                    # Use all numeric columns (ID columns already filtered)
                    agg_dict = {col: ['sum', 'mean'] for col in num_cols}  # ALL numeric columns
                    agg_df = df.groupby(group_col).agg(agg_dict).round(2)
                    
                    # Flatten MultiIndex columns
                    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
                    agg_df = agg_df.reset_index()
                
                    # Sort by first _sum column
                    sum_cols = [c for c in agg_df.columns if '_sum' in c]
                    if sum_cols:
                        agg_df = agg_df.sort_values(sum_cols[0], ascending=False)
                    
                    analysis += agg_df.head(50).to_string(index=False) + "\n"
                    
                    # Cache aggregate result
                    agg_text = analysis[analysis.rfind("=== AGGREGATED"):]
                    set_cached_aggregate(file_path, group_col, agg_text)
                except Exception as e:
                    print(f"Aggregation error: {e}")
                    analysis += "=== SAMPLE DATA ===\n"
                    analysis += df[cat_cols[:2] + num_cols[:6]].head(10).to_string(index=False) + "\n"
        else:
            analysis += "=== SAMPLE DATA ===\n"
            sample_cols = cat_cols[:3] + num_cols[:5]
            if sample_cols:
                try:
                    analysis += df[sample_cols].head(10).to_string(index=False) + "\n"
                except:
                    pass
        
        # Add analytics result if exists
        # Truncate if still too long
        if len(analysis) > 8000:
            analysis = analysis[:8000] + "\n...(truncated)"
        
        # Cache this result
        set_cached_query_result(file_id, query, analysis)
        
        # Dynamic row-level context: include actual data rows so LLM can answer specific questions
        try:
            n_rows = len(df)
            all_cols = df.columns.tolist()
            # Small dataset: include all rows with all columns
            if n_rows <= 200:
                analysis += f"\n=== RAW DATA ({n_rows} rows) ===\n"
                analysis += df.to_string(index=False) + "\n"
            elif n_rows <= 1000:
                # Medium: include all rows but limit columns
                show_cols = all_cols[:20] if len(all_cols) > 20 else all_cols
                analysis += f"\n=== RAW DATA ({n_rows} rows, {len(show_cols)}/{len(all_cols)} cols) ===\n"
                analysis += df[show_cols].to_string(index=False) + "\n"
            else:
                # Large: sample rows
                show_cols = all_cols[:15] if len(all_cols) > 15 else all_cols
                analysis += f"\n=== DATA SAMPLE (100/{n_rows} rows) ===\n"
                analysis += df[show_cols].head(100).to_string(index=False) + "\n"
        except Exception as e:
            print(f"Row context error: {e}")
        
        # Truncate if too long
        if len(analysis) > 15000:
            analysis = analysis[:15000] + "\n...(truncated)"
        
        # === SECTOR DETECTION ===
        sector_detected = 'unknown'
        if model_id and model_id not in ("schema-v0", "", "none"):
            if model_id in _sector_cache:
                sector_detected = _sector_cache[model_id]
                print(f"[SECTOR] Cache hit: {sector_detected}")
            else:
                try:
                    sector_detected = detect_sector_with_llm(list(df.columns))
                    _sector_cache[model_id] = sector_detected
                    print(f"[SECTOR] Detected: {sector_detected}")
                except Exception as e:
                    print(f"[SECTOR] Error: {e}")
        
        # === VERTICAL AI RUNTIME PIPELINE ===
        vertical_result = None
        if model_id and model_id not in ("schema-v0", "", "none") and user_id:
            try:
                schema_output_for_tools = {}
                if ft_structured:
                    top_class = max(ft_structured['class_distribution'].items(), key=lambda x: x[1]['count'])[0] if ft_structured.get('class_distribution') else "unknown"
                    avg_conf = sum(p['confidence'] for p in ft_structured['predictions'][:100]) / max(len(ft_structured['predictions'][:100]), 1)
                    schema_output_for_tools = {
                        "prediction": top_class,
                        "confidence": round(avg_conf, 4),
                        "class_probabilities": {k: v['percentage']/100 for k, v in ft_structured.get('class_distribution', {}).items()},
                        "total_predictions": ft_structured.get('total_predictions', 0)
                    }
                data_for_tools = df.head(1).to_dict(orient='records')[0] if len(df) > 0 else {}
                # Add sector to schema_output for tools/agents
                schema_output_for_tools['sector'] = sector_detected
                vertical_result = run_vertical_pipeline(model_id, user_id, data_for_tools, schema_output_for_tools)
                if vertical_result:
                    analysis += "\n=== VERTICAL AI RUNTIME RESULTS ===\n"
                    for tr in vertical_result.get('post_inference', []):
                        if tr['status'] == 'success':
                            analysis += f"Tool '{tr['tool']}': {json.dumps(tr['output'])}\n"
                    for ar in vertical_result.get('agent_outputs', []):
                        if ar['status'] == 'success':
                            analysis += f"Agent '{ar['agent']}': {json.dumps(ar['output'])}\n"
                    if vertical_result.get('flags'):
                        analysis += f"Flags: {vertical_result['flags']}\n"
                    if vertical_result.get('final_decision'):
                        analysis += f"Final Decision: {vertical_result['final_decision']}\n"
            except Exception as e:
                print(f"[VERTICAL] Pipeline error: {e}")
                import traceback
                traceback.print_exc()
        
        # Return structured JSON with FULL analysis text so LLM can answer questions
        if ft_structured:
            response = {
                'status': 'success',
                'analysis': analysis,
                'predictions': ft_structured['predictions'],
                'class_distribution': ft_structured['class_distribution'],
                'model': {
                    'id': ft_structured['model_id'],
                    'training_accuracy': ft_structured['training_accuracy'],
                    'classes': ft_structured['classes'],
                    'total_predictions': ft_structured['total_predictions']
                },
                'data_summary': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'numeric_columns': len(df.select_dtypes(include=['number']).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
                }
            }
            response['sector_detected'] = sector_detected
            if vertical_result:
                response['vertical_runtime'] = vertical_result
            return jsonify(response)
        
        response = {'analysis': analysis, 'status': 'success', 'sector_detected': sector_detected}
        if vertical_result:
            response['vertical_runtime'] = vertical_result
        return jsonify(response)
    except Exception as e:
        print("="*80)
        print("❌ TRAINING EXCEPTION")
        print("="*80)
        import traceback
        traceback.print_exc()
        print(f"best_acc at exception: {best_acc if 'best_acc' in locals() else 'UNDEFINED'}")
        print("="*80)
        return jsonify({'analysis': f'Error: {e}', 'status': 'error'})

@app.route('/finetune', methods=['POST'])
def finetune(bypass_queue=False):
    # ============================================================
    from training_queue import training_queue
    import uuid
    import threading
    import tempfile
    
    query_id_from_form = request.form.get('query_id', None)

    # Reset training_progress immediately so polling returns "training" not old "completed"
    training_progress.update({"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "training", "eta": "starting...", "start_time": time.time()})
    if not query_id_from_form:
        query_id_from_form = str(uuid.uuid4())
    
    if not bypass_queue:
        queue_check = training_queue.get_status()
        active = queue_check['active_trainings']
        max_concurrent = queue_check['max_concurrent']
    else:
        active = 0
        max_concurrent = 999
    
    if active >= max_concurrent and not bypass_queue:
        print(f"[Queue] Server busy ({active}/{max_concurrent}), queueing task {query_id_from_form}")
        
        session = get_session(query_id_from_form)
        session.update({
            "epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0,
            "status": "queued", "eta": "0%", "start_time": time.time(),
            "query_id": query_id_from_form, "queued": True, "queue_position": 0
        })
        save_session(query_id_from_form, session)
        
        form_data_copy = dict(request.form)
        form_data_copy['query_id'] = query_id_from_form
        
        temp_files = []
        if 'file' in request.files:
            files = request.files.getlist('file')
            for f in files:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                f.save(temp_file.name)
                temp_file.close()
                temp_files.append((temp_file.name, f.filename))
        
        def queued_training():
            try:
                print(f"[Queue] Starting queued training {query_id_from_form}")
                
                sess = get_session(query_id_from_form)
                sess['status'] = 'starting'
                sess['queued'] = False
                save_session(query_id_from_form, sess)
                
                with app.test_request_context(method='POST', data=form_data_copy):
                    from werkzeug.datastructures import FileStorage, MultiDict
                    files_dict = MultiDict()
                    for temp_path, orig_name in temp_files:
                        with open(temp_path, 'rb') as tf:
                            files_dict.add('file', FileStorage(tf, filename=orig_name))
                    
                    with app.request_context(request.environ.copy()):
                        request.files = files_dict
                        request.form = form_data_copy
                        
                        finetune(bypass_queue=True)
                
                for temp_path, _ in temp_files:
                    try: os.unlink(temp_path)
                    except: pass
                    
            except Exception as e:
                print(f"[Queue] Training {query_id_from_form} failed: {e}")
                sess = get_session(query_id_from_form)
                sess['status'] = 'failed'
                sess['error'] = str(e)
                save_session(query_id_from_form, sess)
        
        user_id = request.headers.get('X-User-ID', '')
        result = training_queue.submit(
            task_id=query_id_from_form,
            train_func=queued_training,
            user_id=user_id
        )
        
        return jsonify({
            "task_id": query_id_from_form,
            "status": "queued",
            "queued": True,
            "queue_position": result.get('position', 0),
            "active_trainings": result.get('active', 0),
            "max_concurrent": result.get('max_concurrent', 1),
            "message": result.get('message', 'Training queued')
        }), 202
    
    print(f"[Queue] Server available ({active}/{max_concurrent}), starting immediately")
    
    """Fine-tune model on user data"""
    try:
        print("="*80)
        print("🚀 TRAINING START")
        print("="*80)
        epochs_req = int(request.form.get('epochs', 0))  # 0 = auto
        batch_size_req = int(request.form.get('batch_size', 0))  # 0 = auto
        target_column = request.form.get('target_column', None)
        query_id = request.form.get('query_id', 'default')
        analyze_only = request.form.get('analyze_only', 'false').lower() == 'true'
        
        session = get_session(query_id)
        # print(f"DEBUG FINETUNE START: query_id={query_id}, epochs_req={epochs_req}, analyze_only={analyze_only}")
        # Reset session for new training
        initial_epochs = int(request.form.get('epochs', 5)) or 5
        session.update({"epoch": 0, "epochs": initial_epochs, "accuracy": 0.0, "loss": 0.0, "status": "training", "eta": "0%", "start_time": time.time(), "query_id": query_id, "user_id": request.headers.get("X-User-ID"), "lr": 0.001})
        save_session(query_id, session)
        
        merge_files = request.form.get('merge_files', 'false').lower() == 'true'
        spark_preprocessed = request.form.get('spark_preprocessed', 'false').lower() == 'true'
        spark_merged_path = request.form.get('spark_merged_path', None)
        
        # Spark pre-merged CSV varsa direkt oku, smart_merge atla
        if spark_merged_path and storage_exists(spark_merged_path):
            print(f"[SPARK] Using pre-merged CSV: {spark_merged_path}")
            try:
                df = pd.read_csv(storage_resolve(spark_merged_path) or spark_merged_path)
                print(f"[SPARK] Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
                target_col = auto_select_target(df, target_column)
                if not spark_preprocessed:
                    df, _ = smart_data_cleaning(df)
                    df, _ = smart_time_series_prep(df)
                else:
                    print("[SPARK] Skipping smart_data_cleaning - already preprocessed by Spark")
                merged_file_id = None
                merged_filename = os.path.basename(spark_merged_path) if spark_merged_path else None
                # Training'e direkt git
                goto_training = True
            except Exception as e:
                print(f"[SPARK] Failed to load merged CSV: {e}, falling back to normal flow")
                goto_training = False
        else:
            goto_training = False

        if not goto_training:
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
                    # Multi-row header detection: if first row has many duplicate values, skip it
                    try:
                        first_row = pd.read_csv(temp_file.name, nrows=0).columns.tolist()
                        unique_ratio = len(set(str(c).split('.')[0] for c in first_row)) / max(len(first_row), 1)
                        if unique_ratio < 0.5 and len(first_row) > 3:
                            df_temp = pd.read_csv(temp_file.name, header=1)
                            print(f"Multi-row header detected in {file.filename}, using row 2 as header")
                        else:
                            df_temp = pd.read_csv(temp_file.name)
                    except:
                        df_temp = pd.read_csv(temp_file.name)
                
                dataframes.append(df_temp)
                file_names.append(file.filename)
                os.unlink(temp_file.name)
        
        # Birden fazla dosya varsa smart merge yap
        if not goto_training:
            merged_file_id = None
            if len(dataframes) > 1 and merge_files:
                df = smart_merge_datasets(dataframes, file_names)
                import uuid
                from datetime import datetime
                merged_file_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                merged_filename = f"{merged_file_id[:8]}_merged_all_{timestamp}.csv"
                merged_path = os.path.join('../uploads', merged_filename)
                df.to_csv(merged_path, index=False)
                # Upload merged file to GCS
                try:
                    user_id = request.headers.get("X-User-ID", "system")
                    gcs_key = cloud_storage.user_key(user_id, "uploads", merged_filename)
                    cloud_storage.upload(gcs_key, merged_path)
                    print(f"[STORAGE] Merged file uploaded: {gcs_key}")
                except Exception as me:
                    print(f"[STORAGE] Merged upload failed: {me}")
            else:
                df = dataframes[0]
            
            # Otomatik akıllı target seçimi
            target_col = auto_select_target(df, target_column)
            if not spark_preprocessed:
                df, cleaning_report = smart_data_cleaning(df)
                df, ts_report = smart_time_series_prep(df)
            else:
                print("[SPARK] Skipping smart_data_cleaning - already preprocessed by Spark")
        numeric_df = df.select_dtypes(include=['number'])
        # Agnostik: Tüm sayısal kolonları feature olarak kullan
        col_mapping, feature_cols = smart_column_mapping(numeric_df.columns.tolist(), target_col)
        
        
        # Tüm feature kolonlarını al - numeric yoksa categorical encode et
        if len(feature_cols) == 0:
            cat_cols = [c for c in df.columns if c != target_col and df[c].dtype == 'object']
            if len(cat_cols) == 0:
                return jsonify({"error": "Dataset has no usable features. Please upload a dataset with at least one data column.", "status": "failed"}), 400
            print(f"No numeric features found. Encoding {len(cat_cols)} categorical columns as features.")
            encoded_frames = []
            for col in cat_cols:
                le_feat = LabelEncoder()
                encoded = le_feat.fit_transform(df[col].fillna('__NA__').astype(str))
                encoded_frames.append(encoded)
            X = np.column_stack(encoded_frames).astype(np.float32)
            feature_cols = cat_cols
            print(f"Encoded features shape: {X.shape}")
        else:
            X = df[feature_cols].values.astype(np.float32)
        
        # Eksik değer kontrolü
        has_missing = np.isnan(X).any()
        if has_missing:
            missing_pct = np.isnan(X).mean() * 100
            X = np.nan_to_num(X, nan=0.0)
        
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
        # print(f"Training with {input_dim} features")
        
        # Tam dinamik config
        dyn_cfg = get_dynamic_config(len(X), input_dim, n_classes)
        session["lr"] = dyn_cfg['lr']
        save_session(query_id, session)
        
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
        
        # print(f"Dynamic: d={dyn_cfg['d_model']}, L={dyn_cfg['n_layers']}, lat={dyn_cfg['n_latents']}, bs={dyn_cfg['batch_size']}, ep={dyn_cfg['epochs']}, lr={dyn_cfg['lr']:.4f}")
        # Check if MIRAS is requested
        use_miras = request.form.get('use_miras', 'false').lower() == 'true'
        miras_bias = request.form.get('miras_bias', 'huber')
        miras_retention = request.form.get('miras_retention', 'lq')
        
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
                # print(f"MIRAS Model created with bias={miras_bias}, retention={miras_retention}")
            else:
                ft_model = TabularFoundationModel(ft_config)
            ft_model = ft_model.to(device)
            for m in ft_model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.momentum = 0.01
            if hasattr(torch, "compile") and not torch.cuda.is_available():
                try:
                    ft_model = torch.compile(ft_model, mode="default", backend="eager")
                except:
                    pass
                try:
                    ft_model = torch.quantization.quantize_dynamic(ft_model, {nn.Linear}, dtype=torch.qint8)
                except:
                    pass
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
            redis_client.setex(f"training:{query_id}", 86400, json.dumps(session))
        except Exception as e:
            print(f"[REDIS] async write failed: {e}")
        session["start_time"] = time.time()
        session["epochs"] = 0  # Will be updated during training
        session["epoch"] = 0
        session["accuracy"] = 0.0
        session["loss"] = 0.0
        session["eta"] = "calculating..."
        if "query_id" in dir() and query_id:
            save_session(query_id, session)
        else:
            training_progress.update(session)
        
        # Süre optimizasyonu - epoch başına max sample
        if len(X) > 10000:
            max_samples_per_epoch = 10000
        else:
            max_samples_per_epoch = len(X)
        # === MEMORY GUARD ===
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            mem = psutil.virtual_memory()
            if mem.percent > 85:
                raise MemoryError(f'Insufficient memory: {mem.percent}% used')
        if len(X) > 1000000:
            print(f'[MEMORY GUARD] Sampling from {len(X)} to 1000000 rows')
            idx = np.random.choice(len(X), 1000000, replace=False)
            X = X[idx]; y = y[idx]
        print(f"Starting training loop: X.shape={X.shape}, epochs={epochs}, batch_size={batch_size}, samples_per_epoch={max_samples_per_epoch}")
        
        best_acc = 0
        best_state = None
        patience = dyn_cfg['patience']
        no_improve = 0
        max_epochs = 200  # Maksimum epoch limiti
        training_timeout = time.time() + 1800  # 30 min max
        current_epoch = 0
        
        # DataLoader ile paralel data loading
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(torch.FloatTensor(X[:max_samples_per_epoch]), torch.LongTensor(y[:max_samples_per_epoch]))
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_mem = torch.cuda.is_available()
        prefetch = 2 if num_workers > 0 else None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, prefetch_factor=prefetch)
        
        # Akilli epoch tahmini: data ozellikleri + ogrenme zorlugu
        import math
        rows = len(X)
        # Veri/parametre orani - model ne kadar kolay ezberler
        params_per_class = input_dim * 64 + 64 * n_classes  # yakласіk model parametresi
        data_ratio = rows / max(1, params_per_class)  # cok data = hizli ogrenme
        # Sinif dengesi ve karmasikligi
        class_factor = math.log2(max(2, n_classes))  # 2 sinif=1, 4=2, 8=3
        # Feature zorlugu
        feat_factor = math.log10(max(2, input_dim))  # 10 feat=1, 100=2, 1000=3
        # Temel tahmin
        if data_ratio > 2:  # cok data, az parametre - hizli
            base = int(5 + class_factor * 2)
        elif data_ratio > 0.5:  # dengeli
            base = int(10 + class_factor * 3 + feat_factor * 2)
        elif data_ratio > 0.1:  # az data
            base = int(20 + class_factor * 5 + feat_factor * 3)
        else:  # cok az data, cok parametre
            base = int(40 + class_factor * 8 + feat_factor * 5)
        # Patience ekle (early stop margin) ve sinirla
        estimated_epochs = 0  # Unknown until training progresses
        session["epochs"] = estimated_epochs
        if "query_id" in dir() and query_id:
            save_session(query_id, session)
        else:
            training_progress.update(session)
        print(f"📊 Training loop starting: max_epochs={max_epochs}, best_acc={best_acc}")
        while current_epoch < max_epochs and time.time() < training_timeout:
            ft_model.train()
            total_loss = 0
            correct = 0
            batches = 0
            all_preds = []
            all_labels = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
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
                        mcm_loss = out.get('mcm_loss', torch.tensor(0.0, device=device))
                        miras_loss = out.get('miras_loss', torch.tensor(0.0, device=device))
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
                    mcm_loss = out.get('mcm_loss', torch.tensor(0.0, device=device))
                    miras_loss = out.get('miras_loss', torch.tensor(0.0, device=device))
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
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())
                batches += 1
            
            current_epoch += 1
            scheduler.step()
            acc = 100 * correct / min(len(X), max_samples_per_epoch)
            avg_loss = total_loss / max(batches, 1)
            
            # Calculate precision, recall, f1
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                avg_mode = 'binary' if len(set(all_labels)) == 2 else 'weighted'
                precision = precision_score(all_labels, all_preds, average=avg_mode, zero_division=0)
                recall = recall_score(all_labels, all_preds, average=avg_mode, zero_division=0)
                f1 = f1_score(all_labels, all_preds, average=avg_mode, zero_division=0)
            except:
                precision = acc / 100
                recall = acc / 100
                f1 = acc / 100
            
            session["precision"] = round(precision, 4)
            session["recall"] = round(recall, 4)
            session["f1_score"] = round(f1, 4)
            
            # Always update best_acc on first epoch or if improved
            if best_acc == 0 or acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() if hasattr(v, "clone") else v for k, v in ft_model.state_dict().items()}
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
            # Epoch sayısı epochs'u geçtiyse güncelle
            if current_epoch >= session["epochs"]:
                session["epochs"] = current_epoch + 1
            session["accuracy"] = acc
            session["loss"] = avg_loss
            session["eta"] = eta
            session["lr"] = optimizer.param_groups[0]["lr"] if optimizer else session.get("lr", 0.001)
            if "query_id" in dir() and query_id:
                save_session(query_id, session)
            else:
                training_progress.update(session)
            
            if current_epoch % 10 == 0 and torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"Epoch {current_epoch}: Acc={acc:.1f}% (best={best_acc:.1f}%)")
            import time as _t; _t.sleep(0.1)
            
            if best_acc >= 99.0:
                print(f"🎉 %99+ accuracy - MÜKEMMEL!")
                session["epochs"] = current_epoch
                session["epoch"] = current_epoch
                session["accuracy"] = best_acc
                if "query_id" in dir() and query_id:
                    save_session(query_id, session)
                else:
                    training_progress.update(session)
                break
            
            if best_acc >= 99.0 and no_improve >= patience:  # %99 hedef
                print(f"✅ Early stop at {best_acc:.1f}% (no improve for {patience} epochs)")
                break
            
            if best_acc < 95.0 and no_improve >= patience * 2:
                print(f"⚠️ Early stop at {best_acc:.1f}% (no improve for {patience * 3} epochs, < 95%)")
                break
            
            if current_epoch >= max_epochs:
                print(f"⚠️ Max epoch ({max_epochs}) - best: {best_acc:.1f}%")
                break
            if time.time() >= training_timeout:
                print(f"⏰ Training timeout (30min) at epoch {current_epoch} - best: {best_acc:.1f}%")
                break
        
        if best_state:
            ft_model.load_state_dict(best_state)
        
        session["status"] = "completed"
        session["epochs"] = current_epoch
        session["epoch"] = current_epoch
        session["accuracy"] = best_acc
        # Get user email from session/database
        # Get user email from DB via query_id
        try:
            import psycopg2
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cur = conn.cursor()
            cur.execute("SELECT u.email, u.name FROM users u JOIN queries q ON u.id = q.user_id WHERE q.id = %s", (query_id,))
            result = cur.fetchone()
            if result:
                print(f"Sending completion email to {result[0]}")
                pass  # Email disabled here - Go handles it
                # send_training_email(result[0], "completed", "model", best_acc, user_name=result[1] if len(result) > 1 else None)
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Email send failed: {e}")
        session["accuracy"] = best_acc
        session["epoch"] = current_epoch
        if "query_id" in dir() and query_id:
            save_session(query_id, session)
        else:
            training_progress.update(session)
        
        from datetime import datetime; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ft_path = f'../checkpoints/model_finetuned_{timestamp}.pt'
        
        # Get merged filename for source_file_id
        try:
            merged_filename_for_ckpt = merged_filename
        except NameError:
            merged_filename_for_ckpt = None
        
        import tempfile
        tmp_ckpt = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
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
            'config': ft_config,
            'source_file_id': merged_filename_for_ckpt
        }, tmp_ckpt.name)
        tmp_ckpt.close()
        ft_storage_key = cloud_storage.user_key('system', 'checkpoints', f'model_finetuned_{timestamp}.pt')
        try:
            cloud_storage.upload(ft_storage_key, tmp_ckpt.name)
            print(f"[STORAGE] Checkpoint uploaded: {ft_storage_key}")
        except Exception as se:
            print(f"[STORAGE] Checkpoint upload failed: {se}")
            import shutil
            os.makedirs(os.path.dirname(ft_path), exist_ok=True)
            shutil.copy2(tmp_ckpt.name, ft_path)
        finally:
            os.unlink(tmp_ckpt.name)
        
        
        # Temp dosya varsa sil (tek dosya modunda)
        try:
            if 'temp_file' in dir() and temp_file and hasattr(temp_file, 'name'):
                os.unlink(temp_file.name)
        except:
            pass
        
        model_id = f"model_finetuned_{timestamp}"
        session["model_id"] = model_id
        if "query_id" in dir() and query_id:
            save_session(query_id, session)
        else:
            training_progress.update(session)
        
        # Sector tahmini yap
        ft_model.eval()
        with torch.inference_mode():
            sample_X = torch.FloatTensor(X[:min(100, len(X))]).to(device)
            out = ft_model(sample_X)
            sector_probs = torch.softmax(out['sector'], dim=1)
            sector_conf = sector_probs.max(1).values.mean().item() * 100
            dominant_sector = sector_probs.mean(0).argmax().item()
        
        # Calculate training duration
        training_duration = 0
        if session.get("start_time"):
            training_duration = int(time.time() - session["start_time"])
        
        print(f"✅ TRAINING COMPLETE: best_acc={best_acc:.2f}%, epochs={current_epoch}")
        return jsonify({
            "status": "success",
            "accuracy": float(best_acc),
            "loss": float(avg_loss),
            "precision": session.get("precision", 0),
            "recall": session.get("recall", 0),
            "f1_score": session.get("f1_score", 0),
            "epochs": current_epoch,
            "requested_epochs": epochs,
            "n_classes": n_classes,
            "classes": [str(c) for c in le.classes_],
            "model_path": ft_path,
            "model_id": model_id,
            "rows": len(df),
            "training_duration": training_duration,
            "sector": {
                "id": dominant_sector,
                "confidence": round(sector_conf, 1),
                "description": f"Data cluster {dominant_sector}"
            },
            "target_column": target_col,
            "miras_enabled": use_miras if 'use_miras' in dir() else False,
            "n_features": input_dim,
            "merged_file_id": merged_file_id if 'merged_file_id' in locals() and merged_file_id else None
        })
    except Exception as e:
        print("="*80)
        print("❌ TRAINING EXCEPTION")
        print("="*80)
        import traceback
        traceback.print_exc()
        print(f"best_acc at exception: {best_acc if 'best_acc' in locals() else 'UNDEFINED'}")
        print("="*80)
        
        # Try to return partial results if training started
        partial_acc = 0.0
        if 'best_acc' in locals() and best_acc > 0:
            partial_acc = best_acc
        
        # Update DB training_failed flag
        query_id = request.form.get("query_id") or request.args.get("query_id")
        if query_id:
            try:
                import psycopg2
                conn = psycopg2.connect(os.getenv("DATABASE_URL"))
                cur = conn.cursor()
                cur.execute("UPDATE queries SET training_failed = TRUE, is_training = FALSE WHERE id = %s", (query_id,))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as db_ex:
                print(f"DB update failed: {db_ex}")
        
        # Return error with partial accuracy if available
        error_response = {"error": str(e), "status": "failed"}
        if partial_acc > 0:
            error_response["accuracy"] = partial_acc
            error_response["partial_results"] = True
        
        return jsonify(error_response), 500

@app.route('/training/reset', methods=['POST'])
def reset_training_progress():
    global training_progress
    training_progress = {"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "idle", "eta": "0%", "start_time": 0}
    return jsonify({"status": "reset"})

@app.route('/training/progress', methods=['GET'])
def get_training_progress():
    query_id = request.args.get("query_id")
    if query_id:
        try:
            rc = _get_redis()
            if rc:
                data = rc.get(f"training:{query_id}")
                if data:
                    return jsonify(json.loads(data))
        except:
            pass
    _load_sessions()
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
    
    # Create session immediately
    session = get_session(task_id)
    session.update({"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "starting", "eta": "0%", "start_time": time.time(), "query_id": task_id})
    save_session(task_id, session)
    
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
    def run_finetune(bypass_queue=True):
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
                    finetune(bypass_queue=True)
            
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

# Load sessions on startup and mark stale trainings
_load_sessions()
_save_sessions()  # Save after marking stale

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    """API endpoint - kullanıcı dosya + query gönderir, base model analiz eder, LLM yok"""
    try:
        query = request.form.get('query', 'Analyze this data')
        user_id = request.form.get('user_id', '')
        
        # Dosyayı al
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please upload a file using the "file" field',
                'supported_formats': ['csv', 'xlsx', 'xls', 'json'],
                'example': 'curl -X POST /v1/analyze -F "file=@data.csv" -F "query=Analyze this"'
            }), 400
        
        files = request.files.getlist('file')
        if len(files) > 1:
            return jsonify({
                'error': 'Multiple files not supported',
                'message': 'Please upload only one file at a time',
                'files_received': len(files)
            }), 400
        
        file = files[0]
        if not file.filename:
            return jsonify({
                'error': 'Empty filename',
                'message': 'The uploaded file has no name'
            }), 400
            
        filename = file.filename.lower()
        
        # Format kontrolü
        supported_formats = ['.csv', '.xlsx', '.xls', '.json']
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in supported_formats:
            return jsonify({
                'error': 'Unsupported file format',
                'message': f'File format "{file_ext}" is not supported',
                'your_file': file.filename,
                'supported_formats': ['csv', 'xlsx', 'xls', 'json'],
                'example': 'Upload a CSV, Excel, or JSON file'
            }), 400
        
        # Dosyayı oku
        import pandas as pd
        import tempfile
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        file.save(temp_file.name)
        temp_file.close()
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(temp_file.name)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(temp_file.name)
            elif filename.endswith('.json'):
                df = pd.read_json(temp_file.name)
            else:
                return jsonify({
                    'error': 'Unsupported file type',
                    'message': f'Cannot process file: {filename}',
                    'supported_formats': ['csv', 'xlsx', 'xls', 'json']
                }), 400
        except Exception as e:
            return jsonify({
                'error': 'Failed to parse file',
                'message': str(e),
                'hint': 'Make sure your file is properly formatted and not corrupted',
                'your_file': file.filename
            }), 400
        finally:
            try: os.unlink(temp_file.name)
            except: pass
        
        # Base model ile analiz
        analysis = {
            'file_info': {
                'filename': file.filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            },
            'statistics': {},
            'predictions': []
        }
        
        # Numeric columns için istatistik
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numeric_cols[:10]:  # Max 10 column
            analysis['statistics'][col] = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else 0
            }
        
        # Base model prediction (eğer yüklüyse)
        if base_model is not None:
            try:
                # Numeric değerleri al
                numeric_data = df.select_dtypes(include=['int64', 'float64']).values
                if len(numeric_data) > 0:
                    # İlk 100 satır için prediction
                    sample = numeric_data[:min(100, len(numeric_data))]
                    
                    import torch
                    import torch.nn.functional as F
                    
                    for row in sample[:5]:  # İlk 5 satır örnek
                        # Replace NaN/inf with 0
                        import numpy as np
                        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Pad/truncate to 64 features (model expects n_features=64)
                        row = list(row)
                        if len(row) < 64:
                            row = row + [0] * (64 - len(row))
                        elif len(row) > 64:
                            row = row[:64]
                        
                        X = torch.FloatTensor([row]).to(device)
                        
                        with torch.inference_mode():
                            out = base_model(X)
                            if isinstance(out, dict) and 'sector' in out:
                                probs = F.softmax(out['sector'], dim=1)
                                conf, pred = probs.max(1)
                                conf_val = float(conf.item())
                                # Handle NaN confidence
                                if np.isnan(conf_val) or np.isinf(conf_val):
                                    conf_val = 0.0
                                analysis['predictions'].append({
                                    'sector_id': int(pred.item()),
                                    'confidence': conf_val
                                })
            except Exception as e:
                analysis['model_error'] = str(e)
        else:
            analysis['model_status'] = 'Base model not loaded'
        
        # Model Card formatında response
        import uuid
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        # Predictions formatını düzenle
        formatted_predictions = []
        sector_labels = ['finance', 'healthcare', 'technology', 'retail', 'manufacturing', 
                        'energy', 'real_estate', 'transportation', 'education', 'entertainment',
                        'agriculture', 'hospitality', 'construction', 'telecom', 'media',
                        'government', 'nonprofit', 'other']
        
        for pred in analysis.get('predictions', []):
            sector_id = pred.get('sector_id', 0)
            label = sector_labels[sector_id] if sector_id < len(sector_labels) else f"sector_{sector_id}"
            formatted_predictions.append({
                'label': label,
                'confidence': round(pred.get('confidence', 0), 4)
            })
        
        # Detect sector with LLM
        column_names = analysis['file_info'].get('column_names', [])
        sector_detected = detect_sector_with_llm(column_names)
        
        # Vertical AI Runtime
        vertical_config_id = request.form.get('vertical_config_id', '')
        vertical_result = None
        if vertical_config_id:
            try:
                schema_output_for_vertical = {
                    'predictions': formatted_predictions,
                    'sector': sector_detected,
                    'confidence': formatted_predictions[0]['confidence'] if formatted_predictions else 0
                }
                data_for_vertical = df.iloc[0].to_dict() if len(df) > 0 else {}
                vertical_result = run_vertical_pipeline(None, user_id, data_for_vertical, schema_output_for_vertical, vertical_config_id=vertical_config_id)
                print(f"[ANALYZE] Vertical pipeline done: {vertical_config_id}")
            except Exception as ve:
                print(f"[ANALYZE] Vertical error: {ve}")

        response = {
            'status': 'success',
            'request_id': request_id,
            'predictions': formatted_predictions,
            'sector_detected': sector_detected,
            'data_summary': {
                'filename': analysis['file_info']['filename'],
                'rows': analysis['file_info']['rows'],
                'columns': analysis['file_info']['columns'],
                'numeric_columns': len([k for k in analysis.get('statistics', {}).keys()]),
                'column_names': analysis['file_info']['column_names'][:20]
            },
            'statistics': analysis.get('statistics', {})
        }
        if vertical_result:
            response['vertical_runtime'] = vertical_result
        
        return jsonify(response)
        
    except Exception as e:
        print("="*80)
        print("❌ TRAINING EXCEPTION")
        print("="*80)
        import traceback
        traceback.print_exc()
        print(f"best_acc at exception: {best_acc if 'best_acc' in locals() else 'UNDEFINED'}")
        print("="*80)
        return jsonify({'error': str(e), 'status': 'error'}), 500




# ─── Vertical AI Runtime: Execution Engine ───

def load_vertical_runtime(model_id, user_id, vertical_config_id=None):
    """Load active vertical's tools, agents, config for a model or by vertical_config_id"""
    import psycopg2
    db_url = os.environ.get('DATABASE_URL', 'postgresql://schemalabs:schemalabs@localhost:5432/schemalabs')
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Find vertical config — by ID or by model_id
        if vertical_config_id:
            cur.execute("""
                SELECT id, name, config_yaml FROM vertical_configs 
                WHERE id=%s AND enabled=true LIMIT 1
            """, (vertical_config_id,))
        else:
            cur.execute("""
                SELECT id, name, config_yaml FROM vertical_configs 
                WHERE model_id=%s AND user_id=%s AND enabled=true
                ORDER BY created_at DESC LIMIT 1
            """, (model_id, user_id))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return {"tools": [], "agents": [], "config": {}}
        
        vertical_id = row[0]
        config = {}
        try:
            import yaml
            config = yaml.safe_load(row[2]) or {}
        except:
            try: config = json.loads(row[2])
            except: config = {}
        
        # Load tools
        cur.execute("""
            SELECT id, name, code, hook FROM vertical_tools 
            WHERE vertical_id=%s AND user_id=%s AND validation_status='passed'
            ORDER BY execution_order ASC
        """, (vertical_id, user_id))
        tools = [{"id": r[0], "name": r[1], "code": r[2], "hook": r[3]} for r in cur.fetchall()]
        
        # Load agents
        cur.execute("""
            SELECT id, name, code, role, COALESCE(runs_if,''), COALESCE(parallel_with,'') FROM vertical_agents 
            WHERE vertical_id=%s AND user_id=%s AND validation_status='passed'
            ORDER BY pipeline_order ASC
        """, (vertical_id, user_id))
        agents = [{"id": r[0], "name": r[1], "code": r[2], "role": r[3], "runs_if": r[4], "parallel_with": r[5]} for r in cur.fetchall()]
        
        cur.close(); conn.close()
        print(f"[VERTICAL] Loaded vertical '{row[1]}': {len(tools)} tools, {len(agents)} agents")
        return {"tools": tools, "agents": agents, "config": config}
    except Exception as e:
        print(f"[VERTICAL] Load error: {e}")
        return {"tools": [], "agents": [], "config": {}}


def execute_tool(tool_code, data, schema_output, config, timeout=10):
    """Execute a tool with timeout via threading"""
    import threading
    result = [{"status": "error", "output": None, "error": "Timeout"}]
    
    def _run():
        try:
            ns = {}
            exec(tool_code, ns)
            if 'run' in ns:
                r = ns['run'](data, schema_output, config)
                result[0] = {"status": "success", "output": r}
            else:
                result[0] = {"status": "error", "output": None, "error": "No run() found"}
        except Exception as e:
            result[0] = {"status": "error", "output": None, "error": str(e)}
    
    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=timeout)
    return result[0]


def execute_agent(agent_code, data, schema_output, tool_outputs, tools, config, timeout=15):
    """Execute an agent with timeout via threading"""
    import threading
    result = [{"status": "error", "output": None, "error": "Timeout"}]
    
    def _run():
        try:
            ns = {}
            exec(agent_code, ns)
            if 'Agent' not in ns:
                result[0] = {"status": "error", "output": None, "error": "No Agent class"}
                return
            agent = ns['Agent'](config)
            def tool_runner(name, d2, so2, c2):
                for t in tools:
                    if t['name'] == name:
                        r = execute_tool(t['code'], d2, so2, c2 if c2 else config)
                        return r.get('output', {})
                return {}
            def schema_runner(d2): return schema_output
            r = agent.run(data, schema_output, tool_outputs, tool_runner, schema_runner)
            result[0] = {"status": "success", "output": r}
        except Exception as e:
            result[0] = {"status": "error", "output": None, "error": str(e)}
    
    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=timeout)
    return result[0]


def run_vertical_pipeline(model_id, user_id, data, schema_output, vertical_config_id=None):
    """Full pipeline: pre_inference → schema → post_inference → agent → validator"""
    import time as _time
    _start = _time.time()
    runtime = load_vertical_runtime(model_id, user_id, vertical_config_id=vertical_config_id)
    if not runtime['tools'] and not runtime['agents']:
        return None
    
    config = runtime['config']
    result = {
        "pre_inference": [], "post_inference": [], "agent_outputs": [],
        "validator": [], "flags": [],
        "meta": {"tools_executed": 0, "agents_executed": 0, "tools_ran": [], "tools_skipped": [], "agents_ran": [], "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())}
    }
    
    # Pre-inference tools — can transform data
    for t in runtime['tools']:
        if t['hook'] == 'pre_inference':
            print(f"[VERTICAL] Pre-inference: {t['name']}")
            r = execute_tool(t['code'], data, {}, config)
            result['pre_inference'].append({"tool": t['name'], "status": r['status'], "output": r.get('output')})
            result['meta']['tools_executed'] += 1
            result['meta']['tools_ran'].append(t['name'])
            if r['status'] == 'success' and isinstance(r.get('output'), dict):
                data.update(r['output'])
    
    # Post-inference tools with on_tool_failure support
    tool_outputs = {}
    on_tool_failure = 'skip_and_continue'
    if isinstance(config, dict) and isinstance(config.get('tools'), dict):
        on_tool_failure = config['tools'].get('on_tool_failure', 'skip_and_continue')
    for t in runtime['tools']:
        if t['hook'] == 'post_inference':
            print(f"[VERTICAL] Post-inference: {t['name']}")
            r = execute_tool(t['code'], data, schema_output, config)
            result['post_inference'].append({"tool": t['name'], "status": r['status'], "output": r.get('output')})
            result['meta']['tools_executed'] += 1
            if r['status'] == 'success':
                tool_outputs[t['name']] = r.get('output', {})
                result['meta']['tools_ran'].append(t['name'])
            else:
                result['meta']['tools_skipped'].append(t['name'])
                if on_tool_failure == 'abort':
                    print(f"[VERTICAL] Tool {t['name']} failed — aborting")
                    break
    
    # Agents with conditional execution (runs_if support)
    agent_results = {}
    for a in runtime['agents']:
        # Check runs_if condition
        runs_if = a.get('runs_if', '')
        if runs_if:
            try:
                # Evaluate condition against agent_results, e.g. "primary_classifier.output.requires_review == true"
                parts = runs_if.split('.')
                if len(parts) >= 3 and parts[0] in agent_results:
                    val = agent_results[parts[0]]
                    for p in parts[1:]:
                        if p.startswith('output'):
                            continue
                        if isinstance(val, dict):
                            val = val.get(p)
                    condition_parts = runs_if.split('==')
                    if len(condition_parts) == 2:
                        expected = condition_parts[1].strip().strip('"').strip("'")
                        if expected == 'true': expected = True
                        elif expected == 'false': expected = False
                        if val != expected:
                            print(f"[VERTICAL] Agent {a['name']} skipped (runs_if: {runs_if})")
                            result['meta']['agents_ran'].append(f"{a['name']}:skipped")
                            continue
            except Exception as e:
                print(f"[VERTICAL] runs_if eval error for {a['name']}: {e}")
        # Check if this agent runs in parallel with another
        parallel_with = a.get('parallel_with', '')
        parallel_agent = None
        if parallel_with:
            parallel_agent = next((ag for ag in runtime['agents'] if ag['name'] == parallel_with), None)
        
        if parallel_agent:
            import concurrent.futures
            print(f"[VERTICAL] Agent: {a['name']} (parallel with {parallel_with})")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(execute_agent, a['code'], data, schema_output, tool_outputs, runtime['tools'], config)
                f2 = executor.submit(execute_agent, parallel_agent['code'], data, schema_output, tool_outputs, runtime['tools'], config)
                r = f1.result(timeout=30)
                r2 = f2.result(timeout=30)
            # Record both
            result['agent_outputs'].append({"agent": a['name'], "role": a['role'], "status": r['status'], "output": r.get('output')})
            agent_results[a['name']] = r.get('output', {})
            result['meta']['agents_executed'] += 1
            result['meta']['agents_ran'].append(a['name'])
            if a['role'] == 'decision_maker' and r['status'] == 'success':
                result['final_decision'] = r.get('output', {}).get('final_decision')
            result['agent_outputs'].append({"agent": parallel_agent['name'], "role": parallel_agent['role'], "status": r2['status'], "output": r2.get('output')})
            agent_results[parallel_agent['name']] = r2.get('output', {})
            result['meta']['agents_executed'] += 1
            result['meta']['agents_ran'].append(parallel_agent['name'])
            if parallel_agent['role'] == 'decision_maker' and r2['status'] == 'success':
                result['final_decision'] = r2.get('output', {}).get('final_decision')
            # Skip the parallel agent when we encounter it later
            runtime['_parallel_done'] = runtime.get('_parallel_done', set())
            runtime['_parallel_done'].add(parallel_agent['name'])
            continue
        
        # Skip if already executed in parallel
        if a['name'] in runtime.get('_parallel_done', set()):
            continue
        
        print(f"[VERTICAL] Agent: {a['name']}")
        r = execute_agent(a['code'], data, schema_output, tool_outputs, runtime['tools'], config)
        result['agent_outputs'].append({"agent": a['name'], "role": a['role'], "status": r['status'], "output": r.get('output')})
        agent_results[a['name']] = r.get('output', {})
        result['meta']['agents_executed'] += 1
        result['meta']['agents_ran'].append(a['name'])
        if a['role'] == 'decision_maker' and r['status'] == 'success':
            result['final_decision'] = r.get('output', {}).get('final_decision')
    
    # Validators
    for t in runtime['tools']:
        if t['hook'] == 'validator':
            print(f"[VERTICAL] Validator: {t['name']}")
            vdata = {"original_data": data, "schema_output": schema_output, "tool_outputs": tool_outputs, "agent_outputs": result['agent_outputs']}
            r = execute_tool(t['code'], vdata, schema_output, config)
            result['validator'].append({"tool": t['name'], "status": r['status'], "output": r.get('output')})
            result['meta']['tools_executed'] += 1
            result['meta']['tools_ran'].append(t['name'])
    
    # Flags based on config threshold
    if isinstance(config, dict) and isinstance(config.get('behavior'), dict):
        threshold = config['behavior'].get('confidence_threshold', 0.7)
        conf = schema_output.get('confidence', 0) if isinstance(schema_output, dict) else 0
        if conf < threshold:
            result['flags'].append({"flagged_for_review": True, "flag_reason": f"Confidence {conf:.2f} below threshold {threshold}", "confidence_below_threshold": True})
    
    result['meta']['runtime_ms'] = int((_time.time() - _start) * 1000)
    
    # Field renaming via config output.field_labels
    if isinstance(config, dict) and isinstance(config.get('output'), dict):
        labels = config['output'].get('field_labels', {})
        if isinstance(labels, dict):
            for old_key, new_key in labels.items():
                if old_key in result:
                    result[new_key] = result.pop(old_key)
    
    print(f"[VERTICAL] Done: {result['meta']['tools_executed']}T {result['meta']['agents_executed']}A in {result['meta']['runtime_ms']}ms")
    return result

# ─── Vertical AI Runtime: Script Validation ───

BLOCKED_IMPORTS = {'torch', 'tensorflow', 'keras', 'sys', 'ctypes', 'pickle', 'subprocess'}
BLOCKED_CALLS = {'exec', 'eval', '__import__', 'compile', 'execfile'}
BLOCKED_ATTRS = {'os.system', 'os.popen', 'os.exec', 'os.spawn', 'os.fork'}
ALLOWED_OS = {'os.environ', 'os.getenv', 'os.path'}
MAX_SCRIPT_SIZE = 512 * 1024  # 512 KB


@app.route('/validate_config', methods=['POST'])
def validate_config():
    """Validate a Vertical AI system config (YAML/JSON)"""
    data = request.get_json()
    config_yaml = data.get('config_yaml', '')
    
    checks = []
    errors = []
    
    # Check 1: Not empty
    if not config_yaml.strip():
        errors.append("Config is empty")
        return jsonify({"status": "failed", "error": "; ".join(errors), "checks": checks})
    checks.append("Config is not empty")
    
    # Check 2: Size limit (64KB)
    if len(config_yaml.encode('utf-8')) > 64 * 1024:
        errors.append("Config exceeds 64KB size limit")
        return jsonify({"status": "failed", "error": "; ".join(errors), "checks": checks})
    checks.append("Size check passed")
    
    # Check 3: Parse as YAML, JSON, or plain text
    parsed = None
    config_type = "text"
    try:
        import yaml
        parsed = yaml.safe_load(config_yaml)
        if isinstance(parsed, dict):
            config_type = "yaml"
            checks.append("YAML/JSON parse")
        else:
            parsed = {"instructions": config_yaml}
            config_type = "text"
            checks.append("Plain text config accepted")
    except:
        try:
            parsed = json.loads(config_yaml)
            if isinstance(parsed, dict):
                config_type = "json"
                checks.append("YAML/JSON parse")
            else:
                parsed = {"instructions": config_yaml}
                config_type = "text"
                checks.append("Plain text config accepted")
        except:
            parsed = {"instructions": config_yaml}
            config_type = "text"
            checks.append("Plain text config accepted")
    
    # Check 4: Required fields (only for YAML/JSON)
    if config_type != "text" and 'name' not in parsed:
        checks.append("Tip: add a 'name' field for better organization")
    elif config_type != "text":
        checks.append(f"Name field found: '{parsed['name']}'")
    
    # Check 5: Behavior section (recommended)
    if 'behavior' in parsed:
        behavior = parsed['behavior']
        if isinstance(behavior, dict):
            checks.append(f"Behavior section found ({len(behavior)} rules)")
        else:
            errors.append("'behavior' must be a dict/object")
    else:
        checks.append("No behavior section (optional)")
    
    # Check 6: Output format (optional)
    if 'output_format' in parsed:
        checks.append("Output format section found")
    
    # Check 7: No dangerous content
    dangerous = ['__import__', 'exec(', 'eval(', 'subprocess', 'os.system']
    for d in dangerous:
        if d in config_yaml:
            errors.append(f"Config contains suspicious content: '{d}'")
    if not errors or (len(errors) == 0):
        checks.append("Security check passed")
    
    status = "failed" if errors else "passed"
    return jsonify({
        "status": status,
        "error": "; ".join(errors) if errors else "",
        "checks": checks
    })

@app.route('/validate_script', methods=['POST'])
def validate_script():
    """Validate a Python tool or agent script before accepting it."""
    import ast
    
    data = request.get_json()
    code = data.get('code', '')
    script_type = data.get('script_type', 'tool')  # tool or agent
    hook = data.get('hook', 'post_inference')
    
    checks = []
    errors = []
    
    # Check 1: Size limit
    if len(code.encode('utf-8')) > MAX_SCRIPT_SIZE:
        errors.append(f"Script exceeds maximum size of {MAX_SCRIPT_SIZE // 1024}KB")
    else:
        checks.append("Size check passed")
    
    # Check 2: Syntax validation
    try:
        tree = ast.parse(code)
        checks.append("Syntax check passed")
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return jsonify({"status": "failed", "error": "; ".join(errors), "checks": checks})
    
    # Check 3: Security scan (AST)
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split('.')[0]
                if base_module in BLOCKED_IMPORTS:
                    errors.append(f"Line {node.lineno}: blocked import '{alias.name}'")
                if base_module == 'os' and alias.name not in ALLOWED_OS:
                    errors.append(f"Line {node.lineno}: restricted import '{alias.name}' (only os.environ allowed)")
        
        if isinstance(node, ast.ImportFrom):
            base_module = (node.module or '').split('.')[0]
            if base_module in BLOCKED_IMPORTS:
                errors.append(f"Line {node.lineno}: blocked import from '{node.module}'")
            if base_module == 'subprocess':
                errors.append(f"Line {node.lineno}: blocked import from 'subprocess'")
        
        # Check dangerous calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                errors.append(f"Line {node.lineno}: blocked call '{node.func.id}()'")
            if isinstance(node.func, ast.Attribute):
                full_call = ""
                if isinstance(node.func.value, ast.Name):
                    full_call = f"{node.func.value.id}.{node.func.attr}"
                if full_call in BLOCKED_ATTRS:
                    errors.append(f"Line {node.lineno}: blocked call '{full_call}()'")
                if full_call.startswith('os.') and full_call not in ALLOWED_OS:
                    if node.func.attr not in ('environ', 'getenv', 'path'):
                        errors.append(f"Line {node.lineno}: blocked os call '{full_call}()'")
        
        # Check open() with write mode
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
            if len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and 'w' in str(node.args[1].value):
                    errors.append(f"Line {node.lineno}: file write not allowed")
    
    if not errors:
        checks.append("Security scan passed")
    
    # Check 4: Interface validation
    if hook == 'library':
        # Library modules don't need def run — they are imported by other tools
        checks.append("Library module — no interface check required")
    elif script_type == 'tool':
        # Must have def run(data, schema_output, config)
        has_run = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'run':
                args = [a.arg for a in node.args.args]
                if len(args) >= 3 and args[0] == 'data' and args[1] == 'schema_output' and args[2] == 'config':
                    has_run = True
                else:
                    errors.append(f"Tool function 'run' must have signature: run(data, schema_output, config). Found: run({', '.join(args)})")
                    has_run = True
        if not has_run:
            errors.append("Tool must define function: def run(data, schema_output, config)")
        else:
            checks.append("Interface validated (def run found)")
    
    elif script_type == 'agent':
        # Support both: class Agent with run() OR standalone def run()
        has_class_agent = False
        has_class_run = False
        has_func_run = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Agent':
                has_class_agent = True
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'run':
                        has_class_run = True
            if isinstance(node, ast.FunctionDef) and node.name == 'run' and not has_class_agent:
                args = [a.arg for a in node.args.args]
                if len(args) >= 3:
                    has_func_run = True
        if has_class_agent and has_class_run:
            checks.append("Interface validated (class Agent with run method)")
        elif has_func_run:
            checks.append("Interface validated (standalone run function)")
        elif has_class_agent and not has_class_run:
            errors.append("Agent class must define method: def run()")
        else:
            errors.append("Agent must define: def run(data, schema_output, tool_outputs, config) or class Agent with run()")
    
    # Check 5: Dry-run with synthetic data
    if not errors:
        try:
            namespace = {}
            exec(code, namespace)
            
            if hook == 'library':
                checks.append("Library module compiled successfully")
            elif script_type == 'tool' and 'run' in namespace:
                synthetic_data = {"col1": 1.0, "col2": "test", "col3": 100}
                synthetic_schema_output = {"prediction": "class_a", "confidence": 0.85, "class_probabilities": {"class_a": 0.85, "class_b": 0.15}}
                synthetic_config = {"behavior": {"confidence_threshold": 0.75}}
                result = namespace['run'](synthetic_data, synthetic_schema_output, synthetic_config)
                if not isinstance(result, dict):
                    errors.append(f"Tool must return a dict, got {type(result).__name__}")
                else:
                    checks.append(f"Dry-run successful (returned dict with {len(result)} keys)")
            
            elif script_type == 'agent' and 'Agent' in namespace:
                AgentClass = namespace['Agent']
                synthetic_config = {"behavior": {"confidence_threshold": 0.75}}
                agent_instance = AgentClass(synthetic_config)
                synthetic_data = {"col1": 1.0, "col2": "test"}
                synthetic_schema_output = {"prediction": "class_a", "confidence": 0.85, "class_probabilities": {"class_a": 0.85, "class_b": 0.15}}
                synthetic_tool_outputs = {"sample_tool": {"score": 0.75, "tier": "standard"}}
                def mock_tool_runner(name, data, schema_output, config): return {}
                def mock_schema_runner(data): return synthetic_schema_output
                result = agent_instance.run(synthetic_data, synthetic_schema_output, synthetic_tool_outputs, mock_tool_runner, mock_schema_runner)
                if not isinstance(result, dict):
                    errors.append(f"Agent must return a dict, got {type(result).__name__}")
                elif 'final_decision' not in result:
                    errors.append("Agent must return dict with 'final_decision' key")
                else:
                    checks.append(f"Dry-run successful (returned dict with {len(result)} keys)")
                    checks.append("Note: tool_outputs is dict format: {tool_name: output_dict}")
        
        except Exception as e:
            errors.append(f"Dry-run failed: {str(e)}")
    
    status = "failed" if errors else "passed"
    return jsonify({
        "status": status,
        "error": "; ".join(errors) if errors else "",
        "checks": checks
    })



# ─── Language Layer Endpoints ───

@app.route('/execute_tool_api', methods=['POST'])
def execute_tool_api():
    """Execute a single tool via API - used by Language Layer bridge"""
    try:
        data = request.get_json()
        tool_code = data.get('tool_code', '')
        row_data = data.get('row_data', {})
        schema_output = data.get('schema_output', {})
        config = data.get('config', {})
        
        if not tool_code:
            return jsonify({"status": "error", "error": "No tool_code provided"}), 400
        
        result = execute_tool(tool_code, row_data, schema_output, config, timeout=10)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Run prediction on a single row - used by Language Layer bridge"""
    try:
        data = request.get_json()
        model_id = data.get('model_id', '')
        user_id = data.get('user_id', '')
        row_data = data.get('row_data', {})
        model_path = data.get('model_path', '')
        run_pipeline = data.get('run_pipeline', False)
        vertical_config_id = data.get('vertical_config_id', '')
        
        if not model_id:
            return jsonify({"status": "error", "error": "No model_id provided"}), 400
        
        print(f"DEBUG: Received model_path: {model_path}")
        
        # Load model
        import pandas as pd
        df = pd.DataFrame([row_data])
        
        # Get model config
        model_config = get_cached_finetuned_model(model_id, None, model_path=model_path)
        if model_config is None:
            return jsonify({"status": "error", "error": f"Model {model_id} not found"}), 404
        
        model = model_config['model']
        label_encoder = model_config.get('label_encoder')
        feature_cols = model_config.get('feature_cols', [])
        
        # Prepare features
        import numpy as np
        X = df.reindex(columns=feature_cols, fill_value=0).values.astype(np.float32)
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(X_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
        
        prediction = str(pred_idx)
        if label_encoder and hasattr(label_encoder, 'inverse_transform'):
            prediction = label_encoder.inverse_transform([pred_idx])[0]
        
        confidence = float(probs[pred_idx])
        class_probs = {str(i): float(p) for i, p in enumerate(probs)}
        if label_encoder and hasattr(label_encoder, 'classes_'):
            class_probs = {str(c): float(probs[i]) for i, c in enumerate(label_encoder.classes_)}
        
        result = {
            "status": "success",
            "schema_prediction": str(prediction),
            "schema_confidence": confidence,
            "class_probabilities": class_probs,
        }
        
        # Run vertical pipeline if requested
        if run_pipeline and vertical_config_id:
            schema_output = {"prediction": str(prediction), "confidence": confidence, "probabilities": class_probs}
            vertical_result = run_vertical_pipeline(model_id, user_id, row_data, schema_output, vertical_config_id=vertical_config_id)
            if vertical_result:
                result["tool_outputs"] = vertical_result.get("post_inference", {})
                result["agent_output"] = vertical_result.get("agent_outputs", {})
                result["flags"] = vertical_result.get("flags", {})
                result["final_decision"] = vertical_result.get("final_decision", "")
                result["vertical_runtime"] = vertical_result
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 6000))
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)


# Email notification
