import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import TabularFoundationModel
import os
import sys
import time
import datetime
import tempfile
import glob
from pathlib import Path
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def smart_column_mapping(df_cols, target_col):
    """Kullanıcı kolonlarını V1 feature_cols'a akıllı eşle"""
    
    v1_features = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                   'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                   'recovery_index', 'response_rate']
    
    semantic_map = {
        'primary_score': ['primary', 'main', 'budget', 'spend', 'amount', 'total', 'sum', 'revenue', 'sales', 'income', 'price', 'value', 'campaign'],
        'secondary_score': ['secondary', 'impressions', 'views', 'visits', 'traffic', 'cost', 'expense', 'quantity', 'count', 'volume'],
        'tertiary_score': ['tertiary', 'clicks', 'actions', 'profit', 'margin', 'balance', 'sessions', 'users'],
        'risk_index': ['risk', 'ctr', 'rate', 'ratio', 'percent', 'percentage', 'score'],
        'severity_level': ['severity', 'cpc', 'cpp', 'cpa', 'cost_per', 'level', 'priority', 'tier'],
        'duration_factor': ['duration', 'time', 'period', 'days', 'hours', 'frequency', 'ad_frequency'],
        'frequency_rate': ['conversions', 'leads', 'signups', 'purchases', 'orders', 'transactions', 'sales_count'],
        'intensity_score': ['conversion_rate', 'conv_rate', 'cr', 'intensity', 'strength', 'power', 'magnitude'],
        'recovery_index': ['reach', 'audience', 'coverage', 'recovery', 'retention', 'return', 'roi'],
        'response_rate': ['engagement', 'interaction', 'response', 'feedback', 'click_rate', 'open_rate', 'ctr_percent']
    }
    
    numeric_cols = [c for c in df_cols if c != target_col]
    
    mapped = {}
    used_cols = set()
    
    for v1_col in v1_features:
        if v1_col in numeric_cols and v1_col not in used_cols:
            mapped[v1_col] = v1_col
            used_cols.add(v1_col)
    
    for v1_col in v1_features:
        if v1_col in mapped:
            continue
        
        keywords = semantic_map.get(v1_col, [])
        for user_col in numeric_cols:
            if user_col in used_cols:
                continue
            
            user_col_lower = user_col.lower().replace('_', ' ').replace('-', ' ')
            
            for kw in keywords:
                if kw in user_col_lower or user_col_lower in kw:
                    mapped[v1_col] = user_col
                    used_cols.add(user_col)
                    break
            
            if v1_col in mapped:
                break
    
    remaining_user = [c for c in numeric_cols if c not in used_cols]
    remaining_v1 = [c for c in v1_features if c not in mapped]
    
    for i, v1_col in enumerate(remaining_v1):
        if i < len(remaining_user):
            mapped[v1_col] = remaining_user[i]
        else:
            mapped[v1_col] = None  # Pad with zeros
    
    return mapped, v1_features


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

class SectorModel(nn.Module):
    def __init__(self, n_sectors=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_sectors)
        )
    def forward(self, x):
        return self.net(x)

class SubsectorModel(nn.Module):
    def __init__(self, n_sectors=50, n_subsectors=50):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, 128)
        self.net = nn.Sequential(
            nn.Linear(10 + 128, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_subsectors)
        )
    def forward(self, x, sector):
        return self.net(torch.cat([x, self.emb(sector)], dim=1))


MODEL_PATH = os.getenv('MODEL_V1_PATH', '../checkpoints/schemalabsai_v1.pt')
SERVER_PORT = int(os.getenv('FLASK_PORT', 6000))

print("=" * 60)
print("SCHEMALABSAI V1 - Loading model...")
print("=" * 60)

if not Path(MODEL_PATH).exists():
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

sector_to_id = checkpoint['sector_to_id']
id_to_sector = checkpoint['id_to_sector']
sector_sub_to_id = checkpoint['sector_sub_to_id']
sector_bases = checkpoint['sector_bases']
X_min = np.array(checkpoint['X_min'])
X_max = np.array(checkpoint['X_max'])
feature_cols = checkpoint['feature_cols']
n_sectors = len(sector_to_id)

midas = MIDAS()
sector_model = SectorModel(n_sectors=n_sectors)
subsector_model = SubsectorModel(n_sectors=n_sectors, n_subsectors=50)

midas.load_state_dict(checkpoint['midas'])
sector_model.load_state_dict(checkpoint['sector_model'])
subsector_model.load_state_dict(checkpoint['subsector_model'])

midas.eval()
sector_model.eval()
subsector_model.eval()

id_to_subsector = {}
for sid, sub_map in sector_sub_to_id.items():
    id_to_subsector[sid] = {v: k for k, v in sub_map.items()}

current_model_name = "schemalabsai_v1"
finetuned_models = {}

print(f"Model loaded: {current_model_name}")
print(f"Sectors: {n_sectors}")
print(f"Server ready on port {SERVER_PORT}")
print("=" * 60)

training_sessions = {}
training_progress = {"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "idle", "eta": "0%", "start_time": 0}

def get_session(query_id):
    if query_id not in training_sessions:
        training_sessions[query_id] = {"epoch": 0, "epochs": 0, "accuracy": 0.0, "loss": 0.0, "status": "idle", "eta": "0%", "start_time": 0, "query_id": query_id}
    return training_sessions[query_id]


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "current_model": current_model_name,
        "sectors": list(sector_to_id.keys()),
        "n_sectors": n_sectors
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        "current_model": current_model_name,
        "model_path": MODEL_PATH,
        "sectors": list(sector_to_id.keys()),
        "n_sectors": n_sectors,
        "feature_cols": feature_cols,
        "version": checkpoint.get('version', 'V1')
    })

@app.route('/sectors', methods=['GET'])
def list_sectors():
    sector_info = []
    for sector, sid in sector_to_id.items():
        subsectors = list(sector_sub_to_id[sid].keys())
        sector_info.append({
            "name": sector,
            "id": sid,
            "subsectors": subsectors,
            "n_subsectors": len(subsectors)
        })
    return jsonify({"sectors": sector_info})

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
            
            with torch.no_grad():
                if mask.mean() < 1.0:
                    X_imp = midas.impute(X_t, mask_t)
                else:
                    X_imp = X_t
                
                sec_logits = sector_model(X_imp)
                sec_probs = F.softmax(sec_logits, dim=1)
                sec_conf, sec_pred = sec_probs.max(1)
                
                sub_logits = subsector_model(X_imp, sec_pred)
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
        with torch.no_grad():
            sec_logits = sector_model(X_t)
            sec_probs = F.softmax(sec_logits, dim=1)
            sec_conf, sec_pred = sec_probs.max(1)
            
            sub_logits = subsector_model(X_t, sec_pred)
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
    """Analyze CSV file with fine-tuned model if available"""
    try:
        data = request.json
        file_id = data.get('file_id', '')
        model_id = data.get('model_id', '')
        print(f"DEBUG /analyze: file_id={file_id}, model_id={model_id}")
        
        uploads_dir = '../uploads'
        file_path = None
        
        if os.path.exists(uploads_dir):
            for f in os.listdir(uploads_dir):
                if len(file_id) >= 8 and f.startswith(file_id[:8]):
                    file_path = os.path.join(uploads_dir, f)
                    break
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'analysis': 'File not found.', 'status': 'error'})
        
        df = pd.read_csv(file_path)
        
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        }
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Check for fine-tuned model
        ft_model_info = None
        ft_checkpoint = None
        if model_id:
            checkpoints_dir = '../checkpoints'
            for f in os.listdir(checkpoints_dir):
                if model_id in f and f.endswith('.pt'):
                    ft_path = os.path.join(checkpoints_dir, f)
                    ft_checkpoint = torch.load(ft_path, map_location='cpu', weights_only=False)
                    ft_model_info = {
                        'class_names': ft_checkpoint.get('class_names', []),
                        'n_classes': ft_checkpoint.get('n_classes', 0),
                        'feature_cols': ft_checkpoint.get('feature_cols', []),
                        'model_type': ft_checkpoint.get('model_type', 'unknown')
                    }
                    break
        
        analysis = "=== DATASET OVERVIEW ===\n"
        analysis += f"Total Rows: {stats['rows']}\n"
        analysis += f"Total Columns: {stats['columns']}\n"
        analysis += f"Numeric Columns: {len(numeric_cols)}\n"
        
        # Add fine-tuned model info if available
        if ft_model_info:
            analysis += f"\n=== FINE-TUNED MODEL INFO ===\n"
            analysis += f"Domain: Finance/Banking\n"
            analysis += f"Target Classes ({ft_model_info['n_classes']}): {', '.join(ft_model_info['class_names'])}\n"
            analysis += f"Features: {', '.join(ft_model_info['feature_cols'])}\n"
            
            # Calculate actual distribution from CSV
            target_col = None
            for col in df.columns:
                if col.lower() in ['sector', 'subsector', 'category', 'class', 'target', 'label']:
                    target_col = col
                    break
            if target_col is None and df.columns[-1] not in numeric_cols:
                target_col = df.columns[-1]
            
            if target_col:
                value_counts = df[target_col].value_counts()
                analysis += f"\n=== DATA DISTRIBUTION ===\n"
                analysis += f"{'Subsector':<25} {'Count':>10} {'Percentage':>12}\n"
                analysis += "-" * 50 + "\n"
                for cls, count in value_counts.items():
                    pct = count / len(df) * 100
                    analysis += f"{str(cls):<25} {count:>10} {pct:>11.1f}%\n"
        
        analysis += "\n=== COLUMN STATISTICS ===\n"
        analysis += f"{'Column':<20} {'Min':>12} {'Max':>12} {'Mean':>12}\n"
        analysis += "-" * 60 + "\n"
        for col in numeric_cols[:10]:
            analysis += f"{col:<20} {df[col].min():>12.2f} {df[col].max():>12.2f} {df[col].mean():>12.2f}\n"
        
        if len(numeric_cols) >= 5:
            X = df[numeric_cols[:10]].fillna(0).values.astype(np.float32)
            if X.shape[1] < 10:
                pad = np.zeros((X.shape[0], 10 - X.shape[1]), dtype=np.float32)
                X = np.hstack([X, pad])
            
            X_norm = (X - X_min) / (X_max - X_min + 1e-8)
            X_t = torch.FloatTensor(X_norm)
            
            with torch.no_grad():
                sec_logits = sector_model(X_t)
                sec_pred = sec_logits.argmax(1)
            
            pred_counts = {}
            for p in sec_pred.numpy():
                name = id_to_sector.get(p, f"sector_{p}")
                pred_counts[name] = pred_counts.get(name, 0) + 1
            
            analysis += "\n=== SECTOR PREDICTIONS ===\n"
            analysis += f"{'Sector':<20} {'Count':>10} {'Percentage':>12}\n"
            analysis += "-" * 45 + "\n"
            for sector, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
                pct = count / len(df) * 100
                analysis += f"{sector:<20} {count:>10} {pct:>11.1f}%\n"
        
        return jsonify({
            'analysis': analysis,
            'stats': stats,
            'status': 'success',
            'fine_tuned_model': ft_model_info
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'analysis': f'Error: {str(e)}', 'status': 'error'})

@app.route('/finetune', methods=['POST'])
def finetune():
    """Fine-tune model on user data"""
    try:
        epochs = int(request.form.get('epochs', 10))
        batch_size = int(request.form.get('batch_size', 256))
        target_column = request.form.get('target_column', None)
        query_id = request.form.get('query_id', 'default')
        analyze_only = request.form.get('analyze_only', 'false').lower() == 'true'
        
        session = get_session(query_id)
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_file.name)
        temp_file.close()
        
        df = pd.read_csv(temp_file.name)
        
        if target_column and target_column in df.columns:
            target_col = target_column
        else:
            for col in df.columns:
                if col.lower() in ['category', 'class', 'label', 'target', 'sector', 'subsector']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]
        
        numeric_df = df.select_dtypes(include=['number'])
        col_mapping, v1_features = smart_column_mapping(numeric_df.columns.tolist(), target_col)
        
        print(f"Column mapping: {col_mapping}")
        
        X_list = []
        for v1_col in v1_features:
            user_col = col_mapping.get(v1_col)
            if user_col and user_col in df.columns:
                X_list.append(df[user_col].values.astype(np.float32))
            else:
                X_list.append(np.zeros(len(df), dtype=np.float32))
        
        X = np.column_stack(X_list)
        
        v1_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                   'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                   'recovery_index', 'response_rate']
        has_v1_features = all(c in df.columns for c in v1_cols[:5])
        
        has_missing = np.isnan(X).any()
        if has_missing:
            if has_v1_features:
                mask = (~np.isnan(X)).astype(np.float32)
                X_filled = np.nan_to_num(X, nan=0.0)
                
                midas.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_filled)
                    mask_t = torch.FloatTensor(mask)
                    X_imputed = midas.impute(X_t, mask_t, n_iter=3)
                    X = X_imputed.numpy()
                
                missing_pct = (1 - mask.mean()) * 100
                print(f"MIDAS imputation: {missing_pct:.1f}% missing data filled")
            else:
                missing_pct = np.isnan(X).mean() * 100
                X = np.nan_to_num(X, nan=0.0)
                print(f"Simple fill: {missing_pct:.1f}% missing data filled with 0")
        numeric_cols = [col_mapping.get(v, v) for v in v1_features]
        
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
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
                "smart_epochs": smart_epochs,
                "smart_batch_size": smart_batch,
                "classes": le.classes_.tolist()
            })
        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        if X.shape[1] < 10:
            pad = np.zeros((X.shape[0], 10 - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        elif X.shape[1] > 10:
            X = X[:, :10]
        
        v1_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                   'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                   'recovery_index', 'response_rate']
        
        # Check if we have V1 features (either directly or via mapping)
        has_v1_features = all(c in df.columns for c in v1_cols[:5]) or (col_mapping and len(col_mapping) >= 5)
        
        input_dim = X.shape[1]
        print(f"Training with {input_dim} features")
        
        ft_config = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 2,
            'schema_layers': 2,
            'n_latents': 64,
            'n_features': input_dim,
            'n_classes': n_classes,
            'vocab_size': 50000,
            'n_types': 10,
            'max_cols': max(128, input_dim + 10)
        }
        print(f"Creating TabularFoundationModel with config: {ft_config}")
        try:
            ft_model = TabularFoundationModel(ft_config)
            print(f"Model created successfully, params: {sum(p.numel() for p in ft_model.parameters())}")
        except Exception as e:
            print(f"ERROR creating model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        optimizer = AdamW(ft_model.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        session["status"] = "training"
        session["start_time"] = time.time()
        training_progress.update(session)
        
        max_samples_per_epoch = min(len(X), 10000)
        print(f"Starting training loop: X.shape={X.shape}, epochs={epochs}, batch_size={batch_size}, samples_per_epoch={max_samples_per_epoch}")
        
        best_acc = 0
        best_state = None
        patience = 10
        no_improve = 0
        max_epochs = 200  # Maksimum epoch limiti
        current_epoch = 0
        
        while current_epoch < max_epochs:
            print(f"Epoch {current_epoch + 1} starting...")
            ft_model.train()
            idx = np.random.permutation(len(X))
            total_loss = 0
            correct = 0
            batches = 0
            
            for i in range(0, min(len(X), max_samples_per_epoch) - batch_size + 1, batch_size):
                batch_idx = idx[i:i+batch_size]
                batch_X = torch.FloatTensor(X[batch_idx])
                batch_y = torch.LongTensor(y[batch_idx])
                
                if np.random.random() > 0.5:
                    noise = torch.randn_like(batch_X) * 0.01
                    batch_X = batch_X + noise
                
                optimizer.zero_grad()
                out = ft_model(batch_X)
                logits = out['output']
                loss = loss_fn(logits, batch_y) + out['midas_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(1) == batch_y).sum().item()
                batches += 1
            
            scheduler.step()
            current_epoch += 1
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
            elif best_acc >= 95.0 and no_improve >= patience:
                session["epochs"] = current_epoch
            else:
                session["epochs"] = current_epoch + 1
            session["accuracy"] = acc
            session["loss"] = avg_loss
            session["eta"] = eta
            training_progress.update(session)
            
            print(f"Epoch {current_epoch}: Acc={acc:.1f}% (best={best_acc:.1f}%)")
            
            if best_acc >= 99.0:
                print(f"🎉 %99+ accuracy - MÜKEMMEL!")
                break
            
            if best_acc >= 95.0 and no_improve >= patience:
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
        training_progress.update(session)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ft_path = f'../checkpoints/model_finetuned_{timestamp}.pt'
        
        torch.save({
            'model_state_dict': ft_model.state_dict(),
            'model_type': 'v1_finetune',
            'scaler': scaler,
            'encoder': le,
            'class_names': [str(c) for c in le.classes_],
            'feature_cols': numeric_cols[:10],
            'n_classes': n_classes,
            'input_dim': input_dim,
            'n_sectors': n_sectors
        }, ft_path)
        
        os.unlink(temp_file.name)
        
        model_id = f"model_finetuned_{timestamp}"
        session["model_id"] = model_id
        training_progress.update(session)
        
        return jsonify({
            "status": "success",
            "accuracy": float(best_acc),
            "epochs": current_epoch,
            "requested_epochs": epochs,
            "n_classes": n_classes,
            "classes": [str(c) for c in le.classes_],
            "model_path": ft_path,
            "model_id": model_id,
            "rows": len(df)
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

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=SERVER_PORT, threads=4)

@app.route('/predict/finetuned', methods=['POST'])
def predict_finetuned():
    try:
        data = request.json
        values = np.array(data['values'], dtype=np.float32)
        model_id = data.get('model_id', '')
        
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        checkpoints_dir = '../checkpoints'
        model_path = None
        for f in os.listdir(checkpoints_dir):
            if model_id in f and f.endswith('.pt'):
                model_path = os.path.join(checkpoints_dir, f)
                break
        
        if not model_path:
            return jsonify({"error": "Model not found"}), 404
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        config = {
            'd_model': 256,
            'n_heads': 8,
            'n_latents': 32,
            'n_features': checkpoint.get('n_features', 10),
            'n_classes': checkpoint.get('n_classes', 10),
            'vocab_size': 50000,
            'n_types': 10,
            'max_cols': 64
        }
        
        model = TabularFoundationModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if values.shape[1] != config['n_features']:
            if values.shape[1] < config['n_features']:
                pad = np.zeros((values.shape[0], config['n_features'] - values.shape[1]), dtype=np.float32)
                values = np.hstack([values, pad])
            else:
                values = values[:, :config['n_features']]
        
        X_t = torch.FloatTensor(values)
        
        with torch.no_grad():
            out = model(X_t)
            logits = out['output']
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)
        
        class_names = checkpoint.get('class_names', [])
        
        results = []
        for i in range(len(values)):
            pred_id = pred[i].item()
            pred_name = class_names[pred_id] if pred_id < len(class_names) else f"class_{pred_id}"
            results.append({
                "class": pred_name,
                "class_id": pred_id,
                "confidence": float(conf[i].item()),
                "probabilities": probs[i].tolist()
            })
        
        if model.online_learning_enabled and 'label' in data:
            labels = torch.LongTensor(data['label'])
            model.self_learn(X_t, labels)
            torch.save({
                'model_state_dict': model.state_dict(),
                'n_features': config['n_features'],
                'n_classes': config['n_classes'],
                'class_names': class_names,
                'feature_cols': checkpoint.get('feature_cols', [])
            }, model_path)
        
        return jsonify({
            "predictions": results,
            "model_id": model_id,
            "status": "success"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
