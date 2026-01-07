import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import time
import datetime
import tempfile
import glob
from pathlib import Path
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# ===================== MODEL CLASSES =====================

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

# ===================== LOAD V1 MODEL =====================

MODEL_PATH = os.getenv('MODEL_V1_PATH', '../checkpoints/schemalabsai_v1.pt')
SERVER_PORT = int(os.getenv('FLASK_PORT', 6000))

print("=" * 60)
print("SCHEMALABSAI V1 - Loading model...")
print("=" * 60)

if not Path(MODEL_PATH).exists():
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Load metadata
sector_to_id = checkpoint['sector_to_id']
id_to_sector = checkpoint['id_to_sector']
sector_sub_to_id = checkpoint['sector_sub_to_id']
sector_bases = checkpoint['sector_bases']
X_min = np.array(checkpoint['X_min'])
X_max = np.array(checkpoint['X_max'])
feature_cols = checkpoint['feature_cols']
n_sectors = len(sector_to_id)

# Initialize models
midas = MIDAS()
sector_model = SectorModel(n_sectors=n_sectors)
subsector_model = SubsectorModel(n_sectors=n_sectors, n_subsectors=50)

# Load weights
midas.load_state_dict(checkpoint['midas'])
sector_model.load_state_dict(checkpoint['sector_model'])
subsector_model.load_state_dict(checkpoint['subsector_model'])

midas.eval()
sector_model.eval()
subsector_model.eval()

# Build reverse mappings
id_to_subsector = {}
for sid, sub_map in sector_sub_to_id.items():
    id_to_subsector[sid] = {v: k for k, v in sub_map.items()}

current_model_name = "schemalabsai_v1"
finetuned_models = {}

print(f"Model loaded: {current_model_name}")
print(f"Sectors: {n_sectors}")
print(f"Features: {feature_cols}")
print(f"Server ready on port {SERVER_PORT}")
print("=" * 60)

# Training sessions
training_sessions = {}
training_progress = {"epoch": 0, "epochs": 0, "accuracy": 0, "loss": 0, "status": "idle", "eta": "", "start_time": 0}

def get_session(query_id):
    if query_id not in training_sessions:
        training_sessions[query_id] = {"epoch": 0, "epochs": 0, "accuracy": 0, "loss": 0, "status": "idle", "eta": "", "start_time": 0, "query_id": query_id}
    return training_sessions[query_id]

# ===================== ROUTES =====================

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
        
        # Ensure 10 features
        if values.shape[1] < 10:
            pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
            values = np.hstack([values, pad])
        elif values.shape[1] > 10:
            values = values[:, :10]
        
        results = []
        
        for i in range(len(values)):
            row = values[i:i+1]
            
            # Try to find best sector base
            best_sector = None
            if sector_hint and sector_hint in sector_bases:
                best_sector = sector_hint
            
            # Normalize
            if best_sector:
                base = np.array(sector_bases[best_sector])
                row_sub = row - base
            else:
                row_sub = row
            
            row_norm = (row_sub - X_min) / (X_max - X_min + 1e-8)
            
            # Check for missing
            mask = ~np.isnan(row_norm)
            row_norm = np.nan_to_num(row_norm, nan=0.0)
            
            X_t = torch.FloatTensor(row_norm)
            mask_t = torch.FloatTensor(mask.astype(np.float32))
            
            with torch.no_grad():
                # Impute if needed
                if mask.mean() < 1.0:
                    X_imp = midas.impute(X_t, mask_t)
                else:
                    X_imp = X_t
                
                # Predict sector
                sec_logits = sector_model(X_imp)
                sec_probs = F.softmax(sec_logits, dim=1)
                sec_conf, sec_pred = sec_probs.max(1)
                
                # Predict subsector
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
        
        # Ensure 10 features
        if values.shape[1] < 10:
            pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
            values = np.hstack([values, pad])
        elif values.shape[1] > 10:
            values = values[:, :10]
        
        sector_hint = data.get('sector', None)
        
        # Sector base subtraction
        if sector_hint and sector_hint in sector_bases:
            base = np.array(sector_bases[sector_hint])
            values = values - base
        
        # Normalize
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
    """Analyze CSV file"""
    try:
        data = request.json
        file_id = data.get('file_id', '')
        
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
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        analysis = "=== DATASET OVERVIEW ===\n"
        analysis += f"Total Rows: {stats['rows']}\n"
        analysis += f"Total Columns: {stats['columns']}\n"
        analysis += f"Numeric Columns: {len(numeric_cols)}\n\n"
        
        # Column statistics
        analysis += "=== COLUMN STATISTICS ===\n"
        analysis += f"{'Column':<20} {'Min':>12} {'Max':>12} {'Mean':>12}\n"
        analysis += "-" * 60 + "\n"
        for col in numeric_cols[:10]:
            analysis += f"{col:<20} {df[col].min():>12.2f} {df[col].max():>12.2f} {df[col].mean():>12.2f}\n"
        
        # Try prediction if enough numeric columns
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
            
            # Count predictions
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
            'status': 'success'
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
        batch_size = int(request.form.get('batch_size', 64))
        target_column = request.form.get('target_column', None)
        query_id = request.form.get('query_id', 'default')
        
        session = get_session(query_id)
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Save temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_file.name)
        temp_file.close()
        
        df = pd.read_csv(temp_file.name)
        
        # Find target column
        if target_column and target_column in df.columns:
            target_col = target_column
        else:
            # Auto-detect
            for col in df.columns:
                if col.lower() in ['category', 'class', 'label', 'target', 'sector', 'subsector']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]
        
        feature_cols_user = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols_user].select_dtypes(include=['number']).columns.tolist()
        
        X = df[numeric_cols].fillna(0).values.astype(np.float32)
        
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        n_classes = len(le.classes_)
        
        # Normalize
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        if X.shape[1] < 10:
            pad = np.zeros((X.shape[0], 10 - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        elif X.shape[1] > 10:
            X = X[:, :10]
        
        # Create fine-tune model (simple classifier)
        ft_model = nn.Sequential(
            nn.Linear(10, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        
        optimizer = AdamW(ft_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        session["status"] = "training"
        session["epochs"] = epochs
        session["start_time"] = time.time()
        training_progress.update(session)
        
        best_acc = 0
        
        for epoch in range(epochs):
            ft_model.train()
            idx = np.random.permutation(len(X))
            total_loss = 0
            correct = 0
            
            for i in range(0, len(X) - batch_size, batch_size):
                batch_idx = idx[i:i+batch_size]
                batch_X = torch.FloatTensor(X[batch_idx])
                batch_y = torch.LongTensor(y[batch_idx])
                
                optimizer.zero_grad()
                logits = ft_model(batch_X)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(1) == batch_y).sum().item()
            
            acc = 100 * correct / len(X)
            if acc > best_acc:
                best_acc = acc
            
            session["epoch"] = epoch + 1
            session["accuracy"] = acc
            session["loss"] = total_loss
            training_progress.update(session)
            
            print(f"Epoch {epoch+1}/{epochs}: Acc={acc:.1f}%")
        
        session["status"] = "completed"
        session["accuracy"] = best_acc
        training_progress.update(session)
        
        # Save finetuned model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ft_path = f'../checkpoints/model_finetuned_{timestamp}.pt'
        
        torch.save({
            'model_state_dict': ft_model.state_dict(),
            'scaler': scaler,
            'encoder': le,
            'class_names': [str(c) for c in le.classes_],
            'feature_cols': numeric_cols[:10],
            'n_classes': n_classes
        }, ft_path)
        
        os.unlink(temp_file.name)
        
        return jsonify({
            "status": "success",
            "accuracy": float(best_acc),
            "epochs": epochs,
            "n_classes": n_classes,
            "classes": [str(c) for c in le.classes_],
            "model_path": ft_path
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
