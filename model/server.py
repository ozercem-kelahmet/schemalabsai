import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value)

app = Flask(__name__)

MODEL_PATH = os.getenv('MODEL_PATH')
SERVER_PORT = int(os.getenv('SERVER_PORT', '6000'))

if not MODEL_PATH:
    print("ERROR: MODEL_PATH not set in .env")
    exit(1)

# Model Definitions
class MIDAS(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, mask):
        return self.decoder(self.encoder(torch.cat([x * mask, mask], dim=1)))
    def impute(self, x, mask, n_iter=3):
        current = x * mask
        for _ in range(n_iter):
            current = x * mask + self.forward(current, mask) * (1 - mask)
        return current

class SubsectorModel(nn.Module):
    def __init__(self, n_sectors=50, n_subsectors=50):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, 128)
        self.net = nn.Sequential(
            nn.Linear(10 + 128, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_subsectors)
        )
    def forward(self, x, sector): return self.net(torch.cat([x, self.emb(sector)], dim=1))

# Load checkpoint
print("=" * 60)
print("SCHEMALABS AI - Loading model...")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

n_sectors = checkpoint['n_sectors']
n_subsectors = checkpoint['n_subsectors']
sector_to_id = checkpoint['sector_to_id']
sector_sub_to_id = checkpoint['sector_sub_to_id']
id_to_sector = checkpoint['id_to_sector']
sector_bases = checkpoint['sector_bases']
X_min = np.array(checkpoint['X_min'], dtype=np.float32)
X_max = np.array(checkpoint['X_max'], dtype=np.float32)
feature_cols = checkpoint['feature_cols']
version = checkpoint.get('version', 'unknown')

# Initialize models
midas = MIDAS(10, 512)
subsector_model = SubsectorModel(n_sectors, n_subsectors)

midas.load_state_dict(checkpoint['midas'])
subsector_model.load_state_dict(checkpoint['subsector_model'])

midas.eval()
subsector_model.eval()

print(f"Model: {Path(MODEL_PATH).stem}")
print(f"Version: {version}")
print(f"Sectors: {n_sectors}, Subsectors: {n_subsectors}")
print(f"Server ready on port {SERVER_PORT}")
print("=" * 60)

def find_sector_by_range(primary_score):
    """Primary score'dan sector bul (range lookup)"""
    for sector, base in sector_bases.items():
        base_val = base[0] if isinstance(base, list) else base
        if base_val <= primary_score <= base_val + 5000:
            return sector
    # En yakın sector'ü bul
    best_sector = None
    min_diff = float('inf')
    for sector, base in sector_bases.items():
        base_val = base[0] if isinstance(base, list) else base
        diff = abs(primary_score - base_val)
        if diff < min_diff:
            min_diff = diff
            best_sector = sector
    return best_sector

def preprocess(values, sector_name):
    """Sector base çıkar + MinMax normalize"""
    values = np.array(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    
    if values.shape[1] < 10:
        pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
        values = np.hstack([values, pad])
    elif values.shape[1] > 10:
        values = values[:, :10]
    
    # Sector base çıkar
    if sector_name and sector_name in sector_bases:
        sector_base = np.array(sector_bases[sector_name], dtype=np.float32)
        values = values - sector_base
    
    # MinMax normalize
    values = (values - X_min) / (X_max - X_min + 1e-8)
    return torch.FloatTensor(values)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": Path(MODEL_PATH).stem,
        "version": version,
        "n_sectors": n_sectors,
        "n_subsectors": n_subsectors,
        "sectors": list(sector_to_id.keys())
    })

@app.route('/sectors', methods=['GET'])
def list_sectors():
    sectors = []
    for name, sid in sector_to_id.items():
        subs = list(sector_sub_to_id.get(sid, {}).keys())
        sectors.append({"name": name, "id": sid, "subsectors": subs})
    return jsonify({"sectors": sectors})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        values = data['values']
        handle_missing = data.get('handle_missing', True)
        
        # Tek sample mı çoklu mu?
        if isinstance(values[0], list):
            batch = values
        else:
            batch = [values]
        
        results = []
        for sample in batch:
            sample = np.array(sample, dtype=np.float32)
            if len(sample) < 10:
                sample = np.pad(sample, (0, 10 - len(sample)))
            
            # 1. Sector'ü range lookup ile bul
            primary_score = sample[0]
            sector_name = find_sector_by_range(primary_score)
            sector_id = sector_to_id[sector_name]
            
            # 2. Preprocess (sector base çıkar + MinMax)
            X = preprocess(sample, sector_name)
            
            # 3. Handle missing with MIDAS
            if handle_missing:
                mask = (X != 0).float()
                if mask.sum() < X.numel():
                    with torch.no_grad():
                        X = midas.impute(X, mask, n_iter=3)
            
            # 4. Subsector prediction
            with torch.no_grad():
                s = torch.LongTensor([sector_id])
                logits = subsector_model(X, s)
                sub_pred = logits.argmax(-1).item()
                sub_conf = torch.softmax(logits, -1).max().item()
            
            # 5. Subsector name bul
            sub_name = f"subsector_{sub_pred}"
            if sector_id in sector_sub_to_id:
                for name, idx in sector_sub_to_id[sector_id].items():
                    if idx == sub_pred:
                        sub_name = name
                        break
            
            results.append({
                "sector": sector_name,
                "subsector": sub_name,
                "confidence": float(sub_conf)
            })
        
        return jsonify({"predictions": results, "status": "success"})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==================== FINE-TUNE ENDPOINTS ====================
import threading
import tempfile
import time as time_module
from finetune import finetune_model, analyze_only

training_state = {
    "status": "idle",
    "epoch": 0,
    "epochs": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "batch": 0,
    "batches": 0,
    "model_path": "",
    "query_id": ""
}

@app.route('/training/progress', methods=['GET'])
def training_progress():
    return jsonify(training_state)

@app.route('/finetune', methods=['POST'])
def finetune_endpoint():
    global training_state
    
    # Get files
    files = []
    for key in request.files:
        files.append(request.files[key])
    
    if len(files) == 0:
        return jsonify({"error": "No files provided"}), 400
    
    # Parameters
    epochs = int(request.form.get('epochs', 100))
    batch_size = int(request.form.get('batch_size', 64))
    learning_rate = float(request.form.get('learning_rate', 0.001))
    query_id = request.form.get('query_id', '')
    analyze_only_flag = request.form.get('analyze_only', 'false') == 'true'
    target_column = request.form.get('target_column', None)
    
    # Save files temporarily
    filepaths = []
    for f in files:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        f.save(tmp.name)
        filepaths.append(tmp.name)
    
    # Analyze only
    if analyze_only_flag:
        result = analyze_only(filepaths, target_column)
        for fp in filepaths:
            try: os.unlink(fp)
            except: pass
        return jsonify(result)
    
    # Training async
    def train_async():
        global training_state
        try:
            def progress_cb(state):
                global training_state
                training_state.update(state)
                training_state['query_id'] = query_id
            
            training_state = {
                "status": "starting",
                "epoch": 0,
                "epochs": epochs,
                "query_id": query_id
            }
            
            output_dir = os.path.join(os.path.dirname(__file__), 'finetuned_models')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"model_{query_id}_{int(time_module.time())}.pt")
            
            result = finetune_model(
                filepaths,
                target_column=target_column,
                max_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                midas=midas,  # server.py'deki midas'ı kullan
                output_path=output_path,
                progress_callback=progress_cb
            )
            
            training_state.update(result)
            
        except Exception as e:
            training_state["status"] = "error"
            training_state["error"] = str(e)
        finally:
            for fp in filepaths:
                try: os.unlink(fp)
                except: pass
    
    thread = threading.Thread(target=train_async)
    thread.start()
    
    return jsonify({
        "status": "started",
        "epochs": epochs,
        "query_id": query_id
    })


# ==================== ANALYZE ENDPOINT ====================
from finetune import FineTuneModel

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze CSV with fine-tuned model"""
    data = request.json
    file_id = data.get('file_id', '')
    model_id = data.get('model_id', '')
    query = data.get('query', '')
    message = data.get('message', '')
    
    # Find file
    uploads_dir = '../uploads'
    file_path = None
    
    if file_id and os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if len(file_id) >= 8 and f.startswith(file_id[:8]):
                file_path = os.path.join(uploads_dir, f)
                break
    
    # Find fine-tuned model
    ft_model_path = None
    ft_model_dir = os.path.join(os.path.dirname(__file__), 'finetuned_models')
    
    if model_id:
        # Check in finetuned_models directory
        if os.path.exists(ft_model_dir):
            for f in os.listdir(ft_model_dir):
                if model_id in f and f.endswith('.pt'):
                    ft_model_path = os.path.join(ft_model_dir, f)
                    break
        # Check if model_id is direct path
        if not ft_model_path and os.path.exists(model_id):
            ft_model_path = model_id
    
    analysis = ""
    stats = {}
    predictions = []
    
    # Analyze file if exists
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
            # Find target column
            target_col = None
            for col in df.columns:
                if col.lower() in ['category', 'label', 'target', 'class', 'outcome', 'result', 'pos_primary']:
                    target_col = col
                    break
            if not target_col:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                target_col = cat_cols[0] if cat_cols else df.columns[-1]
            
            feature_cols = [c for c in df.columns if c != target_col]
            numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Build analysis
            analysis = "=== DATASET OVERVIEW ===\n"
            analysis += f"Total Rows: {stats['rows']}\n"
            analysis += f"Total Columns: {stats['columns']}\n"
            analysis += f"Target Column: {target_col}\n"
            analysis += f"Numeric Features: {len(numeric_cols)}\n\n"
            
            # Target Distribution
            if target_col in df.columns:
                analysis += "=== TARGET DISTRIBUTION ===\n"
                analysis += f"{'Category':<20} {'Count':>10} {'Percentage':>12}\n"
                analysis += "-" * 45 + "\n"
                vc = df[target_col].value_counts()
                for val, count in vc.head(10).items():
                    pct = count / len(df) * 100
                    analysis += f"{str(val):<20} {count:>10} {pct:>11.1f}%\n"
                analysis += "\n"
            
            # Column Statistics
            if numeric_cols:
                analysis += "=== COLUMN STATISTICS ===\n"
                analysis += f"{'Column':<25} {'Min':>10} {'Max':>10} {'Mean':>10}\n"
                analysis += "-" * 60 + "\n"
                for col in numeric_cols[:10]:
                    analysis += f"{col[:25]:<25} {df[col].min():>10.1f} {df[col].max():>10.1f} {df[col].mean():>10.1f}\n"
                analysis += "\n"
        
        except Exception as e:
            analysis += f"Error reading file: {str(e)}\n\n"
    
    # Use fine-tuned model if available
    if ft_model_path and os.path.exists(ft_model_path):
        try:
            ft_ckpt = torch.load(ft_model_path, map_location='cpu', weights_only=False)
            
            n_features = ft_ckpt['n_features']
            n_classes = ft_ckpt['n_classes']
            ft_classes = ft_ckpt['classes']
            feature_cols_ft = ft_ckpt.get('feature_cols', [])
            accuracy = ft_ckpt.get('accuracy', 0)
            quality = ft_ckpt.get('quality', {})
            X_min = np.array(ft_ckpt.get('X_min', [0]*n_features))
            X_max = np.array(ft_ckpt.get('X_max', [1]*n_features))
            
            analysis += "=== FINE-TUNED MODEL INFO ===\n"
            analysis += f"Features: {n_features}\n"
            analysis += f"Classes: {n_classes}\n"
            analysis += f"Training Accuracy: {accuracy*100:.1f}%\n"
            analysis += f"Data Quality Score: {quality.get('realistic_target', 0)*100:.0f}%\n"
            analysis += f"Classes: {', '.join(str(c) for c in ft_classes[:10])}{'...' if len(ft_classes) > 10 else ''}\n\n"
            
            # Make predictions if we have data
            if file_path and os.path.exists(file_path):
                try:
                    # Load model
                    ft_model = FineTuneModel(n_features, n_classes)
                    ft_model.load_state_dict(ft_ckpt['model'])
                    ft_model.eval()
                    
                    # Prepare data
                    X = df[feature_cols_ft].fillna(0).values.astype(np.float32) if all(c in df.columns for c in feature_cols_ft) else df.select_dtypes(include=['number']).fillna(0).values[:, :n_features].astype(np.float32)
                    
                    # Pad/trim
                    if X.shape[1] < n_features:
                        X = np.hstack([X, np.zeros((X.shape[0], n_features - X.shape[1]))])
                    elif X.shape[1] > n_features:
                        X = X[:, :n_features]
                    
                    # Normalize
                    X = (X - X_min) / (X_max - X_min + 1e-8)
                    X = np.nan_to_num(X, nan=0.0)
                    
                    # Predict
                    X_tensor = torch.FloatTensor(X)
                    with torch.no_grad():
                        logits = ft_model(X_tensor)
                        probs = torch.softmax(logits, dim=-1)
                        preds = logits.argmax(-1).numpy()
                        confs = probs.max(-1).values.numpy()
                    
                    # Prediction distribution
                    pred_counts = {}
                    for p in preds:
                        label = ft_classes[p] if p < len(ft_classes) else f"Class_{p}"
                        pred_counts[label] = pred_counts.get(label, 0) + 1
                    
                    analysis += "=== MODEL PREDICTIONS ===\n"
                    analysis += f"{'Predicted':<20} {'Count':>10} {'Percentage':>12}\n"
                    analysis += "-" * 45 + "\n"
                    for label, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
                        pct = count / len(preds) * 100
                        analysis += f"{str(label):<20} {count:>10} {pct:>11.1f}%\n"
                    analysis += f"\nAverage Confidence: {np.mean(confs)*100:.1f}%\n"
                    analysis += f"High Confidence (>80%): {(confs > 0.8).sum() / len(confs) * 100:.1f}%\n\n"
                    
                    predictions = [{"class": ft_classes[p] if p < len(ft_classes) else f"Class_{p}", "confidence": float(c)} for p, c in zip(preds[:100], confs[:100])]
                    
                except Exception as e:
                    analysis += f"Prediction error: {str(e)}\n\n"
        
        except Exception as e:
            analysis += f"Model load error: {str(e)}\n\n"
    
    # Add user query context
    if query or message:
        analysis += f"=== USER QUERY ===\n{query or message}\n\n"
    
    return jsonify({
        'analysis': analysis.strip(),
        'predictions': predictions,
        'stats': stats,
        'status': 'success'
    })

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=SERVER_PORT, threads=4)
