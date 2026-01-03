import warnings
import datetime
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import sys
import os
import time
from pathlib import Path
import glob
sys.path.append('.')

from model import TabularFoundationModel
from config import get_config

app = Flask(__name__)

env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

MODEL_PATH = os.getenv('MODEL_PATH', '../checkpoints/model_12sector.pt')
SERVER_PORT = int(os.getenv('SERVER_PORT', '6000'))

print("=" * 60)
print("SCHEMALABS AI - Loading model...")
print("=" * 60)

if not Path(MODEL_PATH).exists():
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
config = checkpoint['config']
scalers = checkpoint.get('scalers', {})
encoders = checkpoint.get('encoders', {})
class_offsets = checkpoint.get('class_offsets', {})
sectors = checkpoint.get('sectors', [])

model = TabularFoundationModel(config)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

current_model_path = MODEL_PATH
current_model_name = Path(MODEL_PATH).stem

finetuned_models = {}

@app.route("/training/progress", methods=["GET"])
def get_training_progress():
    query_id = request.args.get("query_id")
    if query_id and query_id in training_sessions:
        return jsonify(training_sessions[query_id])
    # Return any active training or idle
    for qid, session in training_sessions.items():
        if session["status"] == "training":
            return jsonify(session)
    return jsonify(training_progress)

@app.route("/training/progress/<query_id>", methods=["GET"])
def get_training_progress_by_id(query_id):
    if query_id in training_sessions:
        return jsonify(training_sessions[query_id])
    return jsonify({"status": "idle", "query_id": query_id})

@app.route("/training/sessions", methods=["GET"])
def get_all_training_sessions():
    return jsonify(training_sessions)

# Multiple training sessions support
training_sessions = {}

def get_session(query_id):
    if query_id not in training_sessions:
        training_sessions[query_id] = {"epoch": 0, "epochs": 0, "accuracy": 0, "loss": 0, "status": "idle", "eta": "", "start_time": 0, "query_id": query_id}
    return training_sessions[query_id]

# Legacy single progress (for backward compatibility)
training_progress = {"epoch": 0, "epochs": 0, "accuracy": 0, "loss": 0, "status": "idle", "eta": "", "start_time": 0}

class_to_sector = {}
class_names = {}
for sector in sectors:
    offset = int(class_offsets[sector])
    for i, cls_name in enumerate(encoders[sector].classes_):
        global_class = offset + i
        class_to_sector[global_class] = sector
        class_names[global_class] = f"{sector}:{cls_name}"

print(f"Model loaded: {current_model_name}")
print(f"Sectors: {len(sectors)}")
print(f"Total classes: {config['n_classes']}")
print(f"Server ready on port {SERVER_PORT}")
print("=" * 60)

def load_finetuned_model(file_id):
    if file_id in finetuned_models:
        return finetuned_models[file_id]
    
    checkpoint_dir = Path('../checkpoints')
    for f in sorted(checkpoint_dir.glob('model_finetuned_*.pt'), reverse=True):
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
            ft_config = ckpt['config']
            ft_model = TabularFoundationModel(ft_config)
            ft_model.load_state_dict(ckpt['model_state_dict'])
            ft_model.eval()
            
            finetuned_models[file_id] = {
                'model': ft_model,
                'checkpoint': ckpt,
                'path': f
            }
            print(f"Loaded finetuned model: {f.name}")
            return finetuned_models[file_id]
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "current_model": current_model_name,
        "sectors": sectors,
        "n_classes": int(config['n_classes'])
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        "current_model": current_model_name,
        "model_path": current_model_path,
        "sectors": sectors,
        "n_classes": int(config['n_classes']),
        "class_names": {str(k): v for k, v in class_names.items()}
    })

@app.route('/sectors', methods=['GET'])
def list_sectors():
    sector_info = []
    for sector in sectors:
        classes = [str(c) for c in encoders[sector].classes_]
        sector_info.append({
            "name": sector,
            "classes": classes,
            "n_classes": len(classes),
            "offset": int(class_offsets[sector])
        })
    return jsonify({"sectors": sector_info})

@app.route('/models/list', methods=['GET'])
def list_models():
    models = []
    checkpoint_files = glob.glob('../checkpoints/*.pt')
    for f in sorted(checkpoint_files, reverse=True):
        filename = os.path.basename(f)
        name = filename.replace('.pt', '').replace('_', ' ').title()
        models.append({
            "name": name,
            "filename": filename,
            "path": f,
            "type": "finetuned" if "finetuned" in filename else "base",
            "is_current": current_model_path == f
        })
    return jsonify({"models": models, "current": current_model_name})

@app.route('/models/switch', methods=['POST'])
def switch_model():
    global model, config, scalers, encoders, class_offsets, sectors
    global current_model_path, current_model_name, class_to_sector, class_names
    
    try:
        data = request.json
        model_path = data.get('model_path')
        
        if not model_path or not Path(model_path).exists():
            return jsonify({"error": "Model not found"}), 404
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        scalers = checkpoint.get('scalers', {})
        encoders = checkpoint.get('encoders', {})
        class_offsets = checkpoint.get('class_offsets', {})
        sectors = checkpoint.get('sectors', [])
        
        model = TabularFoundationModel(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        
        class_to_sector = {}
        class_names = {}
        for sector in sectors:
            offset = int(class_offsets[sector])
            for i, cls_name in enumerate(encoders[sector].classes_):
                global_class = offset + i
                class_to_sector[global_class] = sector
                class_names[global_class] = f"{sector}:{cls_name}"
        
        current_model_path = model_path
        current_model_name = Path(model_path).stem
        
        return jsonify({
            "status": "success",
            "current_model": current_model_name,
            "sectors": sectors,
            "n_classes": int(config['n_classes'])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        values = np.array(data['values'], dtype=np.float32)
        sector = data.get('sector', None)
        
        if sector and sector in scalers:
            values = scalers[sector].transform(values)
        
        if values.shape[1] < 10:
            pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
            values = np.hstack([values, pad])
        elif values.shape[1] > 10:
            values = values[:, :10]
        
        values_tensor = torch.FloatTensor(values)
        
        with torch.no_grad():
            outputs = model(values=values_tensor, continuous=True, task='classification')
        
        logits = outputs['base_output']
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).tolist()
        confidences = [float(probs[i][predictions[i]]) for i in range(len(predictions))]
        
        pred_names = [class_names.get(p, f"class_{p}") for p in predictions]
        pred_sectors = [class_to_sector.get(p, "unknown") for p in predictions]
        
        return jsonify({
            "predictions": predictions,
            "prediction_names": pred_names,
            "prediction_sectors": pred_sectors,
            "confidences": confidences,
            "model_used": current_model_name,
            "status": "success"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict/sector', methods=['POST'])
def predict_sector():
    try:
        data = request.json
        values = np.array(data['values'], dtype=np.float32)
        sector = data['sector']
        
        if sector not in sectors:
            return jsonify({"error": f"Unknown sector: {sector}"}), 400
        
        if sector in scalers:
            values = scalers[sector].transform(values)
        
        if values.shape[1] < 10:
            pad = np.zeros((values.shape[0], 10 - values.shape[1]), dtype=np.float32)
            values = np.hstack([values, pad])
        elif values.shape[1] > 10:
            values = values[:, :10]
        
        values_tensor = torch.FloatTensor(values)
        
        with torch.no_grad():
            outputs = model(values=values_tensor, continuous=True, task='classification')
        
        logits = outputs['base_output']
        offset = int(class_offsets[sector])
        n_classes = len(encoders[sector].classes_)
        sector_logits = logits[:, offset:offset+n_classes]
        
        probs = torch.softmax(sector_logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).tolist()
        confidences = [float(probs[i][predictions[i]]) for i in range(len(predictions))]
        pred_names = [str(encoders[sector].classes_[p]) for p in predictions]
        
        return jsonify({
            "sector": sector,
            "predictions": predictions,
            "prediction_names": pred_names,
            "confidences": confidences,
            "probabilities": [[float(x) for x in row] for row in probs.tolist()],
            "class_names": [str(c) for c in encoders[sector].classes_],
            "model_used": current_model_name,
            "status": "success"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze CSV with fine-tuned model - returns formatted tables"""
    data = request.json
    file_id = data.get('file_id', '')
    query = data.get('query', '')
    
    uploads_dir = '../uploads'
    file_path = None
    
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if len(file_id) >= 8 and f.startswith(file_id[:8]):
                file_path = os.path.join(uploads_dir, f)
                break
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({
            'analysis': 'File not found.',
            'predictions': [],
            'stats': {},
            'status': 'error'
        })
    
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
            if col.lower() in ['category', 'label', 'target', 'class', 'outcome', 'result']:
                target_col = col
                break
        if not target_col:
            target_col = df.columns[-1]
        
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Build analysis with TABLES
        analysis = "=== DATASET OVERVIEW ===\n"
        analysis += f"Total Rows: {stats['rows']}\n"
        analysis += f"Total Columns: {stats['columns']}\n"
        analysis += f"Target Column: {target_col}\n"
        analysis += f"Feature Columns: {', '.join(numeric_cols)}\n\n"
        
        # TABLE 1: Target Distribution
        if target_col in df.columns:
            analysis += "=== TARGET DISTRIBUTION TABLE ===\n"
            analysis += f"{'Category':<15} {'Count':>10} {'Percentage':>12}\n"
            analysis += "-" * 40 + "\n"
            vc = df[target_col].value_counts()
            for val, count in vc.items():
                pct = count / len(df) * 100
                analysis += f"{str(val):<15} {count:>10} {pct:>11.1f}%\n"
            analysis += "\n"
        
        # TABLE 2: Numeric Column Statistics
        analysis += "=== COLUMN STATISTICS TABLE ===\n"
        analysis += f"{'Column':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}\n"
        analysis += "-" * 70 + "\n"
        for col in numeric_cols:
            analysis += f"{col:<20} {df[col].min():>12.2f} {df[col].max():>12.2f} {df[col].mean():>12.2f} {df[col].std():>12.2f}\n"
        analysis += "\n"
        
        # TABLE 3: Group Comparison (if categorical target)
        if target_col in df.columns and df[target_col].dtype == 'object':
            analysis += "=== COMPARISON BY TARGET TABLE ===\n"
            categories = df[target_col].unique()
            
            # Header
            header = f"{'Column':<20}"
            for cat in categories:
                header += f" {str(cat):>15}"
            analysis += header + "\n"
            analysis += "-" * (20 + 16 * len(categories)) + "\n"
            
            # Rows
            for col in numeric_cols[:6]:
                row = f"{col:<20}"
                for cat in categories:
                    avg = df[df[target_col] == cat][col].mean()
                    row += f" {avg:>15.2f}"
                analysis += row + "\n"
            analysis += "\n"
        
        # TABLE 4: Correlation with target
        if target_col in df.columns and df[target_col].dtype == 'object' and len(numeric_cols) > 0:
            df_encoded = df.copy()
            df_encoded[target_col] = pd.factorize(df[target_col])[0]
            
            analysis += "=== CORRELATION WITH TARGET TABLE ===\n"
            analysis += f"{'Column':<20} {'Correlation':>15} {'Strength':>15}\n"
            analysis += "-" * 50 + "\n"
            
            correlations = []
            for col in numeric_cols:
                corr = df_encoded[col].corr(df_encoded[target_col])
                correlations.append((col, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for col, corr in correlations:
                strength = "Strong" if abs(corr) > 0.7 else "Medium" if abs(corr) > 0.4 else "Weak"
                analysis += f"{col:<20} {corr:>15.3f} {strength:>15}\n"
            analysis += "\n"
        
        # Load and use fine-tuned model
        ft_data = load_finetuned_model(file_id)
        
        if ft_data and ft_data['model'] is not None:
            ft_model = ft_data['model']
            ft_checkpoint = ft_data['checkpoint']
            ft_scaler = ft_checkpoint.get('scalers', {}).get('user')
            ft_encoder = ft_checkpoint.get('encoders', {}).get('user')
            ft_classes = ft_checkpoint.get('class_names', [])
            
            analysis += "=== FINE-TUNED MODEL RESULTS ===\n\n"
            
            X = df[numeric_cols].fillna(0).values.astype(np.float32)
            
            if ft_scaler:
                # Match feature count to what scaler expects
                expected_features = ft_scaler.n_features_in_
                current_features = X.shape[1]
                
                if current_features < expected_features:
                    # Pad with zeros
                    pad = np.zeros((X.shape[0], expected_features - current_features), dtype=np.float32)
                    X = np.hstack([X, pad])
                elif current_features > expected_features:
                    # Truncate to expected features
                    X = X[:, :expected_features]
                
                X_scaled = ft_scaler.transform(X)
            else:
                X_scaled = X
            
            if X_scaled.shape[1] < 10:
                pad = np.zeros((X_scaled.shape[0], 10 - X_scaled.shape[1]), dtype=np.float32)
                X_scaled = np.hstack([X_scaled, pad])
            elif X_scaled.shape[1] > 10:
                X_scaled = X_scaled[:, :10]
            
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                outputs = ft_model(values=X_tensor, continuous=True, task='classification')
                logits = outputs['base_output']
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1).numpy()
                confidences = probs.max(dim=-1).values.numpy()
            
            # Model prediction table
            pred_counts = {}
            for p in predictions:
                label = ft_classes[p] if p < len(ft_classes) else f"Class_{p}"
                pred_counts[label] = pred_counts.get(label, 0) + 1
            
            analysis += "Model Prediction Distribution:\n"
            analysis += f"{'Predicted':<15} {'Count':>10} {'Percentage':>12}\n"
            analysis += "-" * 40 + "\n"
            for label, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
                pct = count / len(predictions) * 100
                analysis += f"{label:<15} {count:>10} {pct:>11.1f}%\n"
            analysis += "\n"
            
            # Confidence stats
            analysis += f"Model Confidence: {np.mean(confidences):.1%} average\n"
            analysis += f"High confidence (>80%): {np.sum(confidences > 0.8) / len(confidences) * 100:.1f}%\n\n"
            
            # Accuracy if we have labels
            if target_col in df.columns and df[target_col].dtype == 'object' and ft_encoder:
                try:
                    actual = df[target_col].values
                    actual_encoded = ft_encoder.transform(actual)
                    accuracy = np.mean(predictions == actual_encoded) * 100
                    analysis += f"Model Accuracy: {accuracy:.1f}%\n\n"
                except:
                    pass
        
        return jsonify({
            'analysis': analysis,
            'predictions': pred_counts if 'pred_counts' in dir() else {},
            'stats': stats,
            'status': 'success'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'analysis': f'Error: {str(e)}',
            'predictions': [],
            'stats': {},
            'status': 'error'
        })

@app.route('/finetune', methods=['POST'])
def finetune():
    try:
        import tempfile
        from torch.optim import AdamW
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
        import datetime
        
        epochs = int(request.form.get('epochs', 5))
        batch_size = int(request.form.get('batch_size', 64))
        target_column = request.form.get('target_column', None)
        analyze_only = request.form.get("analyze_only", "false") == "true"
        query_id = request.form.get('query_id', 'default')
        
        # Initialize session for this query
        session = get_session(query_id)
        
        # Collect all files (file, file0, file1, file2, ...)
        files_to_process = []
        temp_files = []
        
        if 'file' in request.files:
            files_to_process.append(request.files['file'])
        
        # Check for numbered files (file0, file1, ...)
        for i in range(20):
            key = f'file{i}'
            if key in request.files:
                files_to_process.append(request.files[key])
        
        if not files_to_process:
            return jsonify({"error": "No file provided"}), 400
        
        # Read all files into DataFrames
        dataframes = []
        for file in files_to_process:
            filename = file.filename.lower()
            
            if filename.endswith('.csv'):
                suffix = '.csv'
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                suffix = '.xlsx'
            elif filename.endswith('.parquet'):
                suffix = '.parquet'
            elif filename.endswith('.json'):
                suffix = '.json'
            else:
                suffix = '.csv'
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            file.save(temp_file.name)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            try:
                if suffix == '.csv':
                    df = pd.read_csv(temp_file.name)
                elif suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(temp_file.name)
                elif suffix == '.parquet':
                    df = pd.read_parquet(temp_file.name)
                elif suffix == '.json':
                    df = pd.read_json(temp_file.name)
                else:
                    df = pd.read_csv(temp_file.name)
                
                dataframes.append(df)
                print(f"Loaded {file.filename}: {len(df)} rows, {len(df.columns)} cols")
            except Exception as e:
                print(f"Error reading file {file.filename}: {e}")
                continue
        
        if not dataframes:
            return jsonify({"error": "Could not read any files"}), 400
        
        # Smart merge all dataframes with JOIN
        merged_csv_path = None
        if len(dataframes) > 1:
            df = dataframes[0]
            
            for other_df in dataframes[1:]:
                # Find join column (id, *_id patterns)
                join_col = None
                df_cols = set(df.columns)
                other_cols = set(other_df.columns)
                common = df_cols.intersection(other_cols)
                
                # Priority: id columns
                for col in common:
                    if col.lower() in ['id', 'user_id', 'order_id', 'product_id', 'customer_id'] or col.lower().endswith('_id'):
                        join_col = col
                        break
                
                if join_col:
                    # LEFT JOIN on common id column
                    df = df.merge(other_df, on=join_col, how='outer', suffixes=('', '_y'))
                    # Remove duplicate columns
                    df = df[[c for c in df.columns if not c.endswith('_y')]]
                    print(f"Joined on '{join_col}', result: {len(df)} rows, {len(df.columns)} cols")
                else:
                    # No join column - concat columns side by side (align by index)
                    if len(df) == len(other_df):
                        for col in other_df.columns:
                            if col not in df.columns:
                                df[col] = other_df[col].values
                    else:
                        # Different lengths - outer concat
                        df = pd.concat([df.reset_index(drop=True), other_df.reset_index(drop=True)], axis=1)
                        df = df.loc[:, ~df.columns.duplicated()]
                    print(f"Concat columns, result: {len(df)} rows, {len(df.columns)} cols")
            
            # Fill NaN with 0 for numeric, empty string for others
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('')
            
            # Save merged CSV
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("./uploads", exist_ok=True)
            merged_csv_path = f"./uploads/merged_{timestamp}.csv"
            df.to_csv(merged_csv_path, index=False)
            print(f"Merged {len(dataframes)} files -> {merged_csv_path}, total: {len(df)} rows, {len(df.columns)} cols")
        else:
            df = dataframes[0]
        
        if target_column and target_column in df.columns:
            target_col = target_column
        else:
            best_target = None
            best_score = 0
            for col in df.columns:
                unique_count = df[col].nunique()
                is_categorical = df[col].dtype == "object" or unique_count < 50
                if is_categorical and 2 <= unique_count <= 100:
                    score = 100 - unique_count
                    if df[col].dtype == "object":
                        score += 50
                    if col.lower() in ["category", "class", "label", "target", "type", "status", "result"]:
                        score += 100
                    if score > best_score:
                        best_score = score
                        best_target = col
            target_col = best_target if best_target else df.columns[-1]
            print(f"Auto-detected target: {target_col} (score={best_score})")
        
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].select_dtypes(include=['number']).fillna(0).values.astype(np.float32)
        
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        n_classes = len(le.classes_)

        print(f"DEBUG: target_col={target_col}, n_classes={n_classes}, unique_targets={len(df[target_col].unique())}")
        # Smart hyperparameter calculation
        user_epochs = epochs
        user_batch_size = batch_size
        
        # Calculate data characteristics
        n_samples = len(X)
        n_features = X.shape[1]
        n_files = len(dataframes) if "dataframes" in dir() else 1
        
        # Check for duplicate/similar rows
        unique_rows = len(np.unique(X, axis=0))
        duplicate_ratio = 1 - (unique_rows / n_samples) if n_samples > 0 else 0
        

        samples_per_class = n_samples / max(n_classes, 1)
        
        if user_epochs == 5:
            if n_classes > 100:
                epochs = 50
            elif n_classes > 50:
                epochs = 30
            elif n_classes > 20:
                epochs = 20
            elif n_classes > 10:
                epochs = 15
            else:
                epochs = 10
            
            if samples_per_class < 10:
                epochs = int(epochs * 2)
            elif samples_per_class < 50:
                epochs = int(epochs * 1.5)
            
            if duplicate_ratio > 0.3:
                epochs = int(epochs * 1.5)
            
            epochs = min(epochs, 100)
        
        if user_batch_size == 64:
            if n_samples < 200:
                batch_size = 8
            elif n_samples < 500:
                batch_size = 16
            elif n_samples < 2000:
                batch_size = 32
            else:
                batch_size = 64
        
        base_lr = 2e-3
        if samples_per_class < 20:
            learning_rate = base_lr * 2
        elif n_samples > 10000:
            learning_rate = base_lr * 0.5
        else:
            learning_rate = base_lr
        
        print(f"Smart params: epochs={epochs}, batch={batch_size}, lr={learning_rate:.6f}, samples/class={samples_per_class:.1f}")

        if analyze_only:
            for tf in temp_files:
                try:
                    os.unlink(tf)
                except:
                    pass
            return jsonify({
                "status": "analyzed",
                "smart_epochs": epochs,
                "smart_batch_size": batch_size,
                "smart_learning_rate": learning_rate,
                "duplicate_ratio": duplicate_ratio,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "n_files": len(dataframes)
            })
        
        scaler = MinMaxScaler(feature_range=(0, 100))
        X = scaler.fit_transform(X)
        
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        X = (X - X_mean) / X_std
        
        # Normalize like base model training
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        X = (X - X_mean) / X_std
        
        if X.shape[1] < 10:
            pad = np.zeros((X.shape[0], 10 - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, pad])
        elif X.shape[1] > 10:
            X = X[:, :10]
        
        finetune_config = config.copy()
        finetune_config['n_classes'] = n_classes
        
        finetuned_model = TabularFoundationModel(finetune_config)
        
        base_state = model.state_dict()
        new_state = finetuned_model.state_dict()
        
        for key in base_state:
            if 'classification_head' not in key and key in new_state:
                if base_state[key].shape == new_state[key].shape:
                    new_state[key] = base_state[key]
        
        finetuned_model.load_state_dict(new_state)
        

        optimizer = AdamW(finetuned_model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        finetuned_model.train()
        session["status"] = "training"
        session["epoch"] = 0
        session["accuracy"] = 0
        session["loss"] = 0
        session["epochs"] = epochs
        session["eta"] = "..."
        session["start_time"] = time.time()
        # Also update legacy progress for backward compatibility
        training_progress.update(session)
        final_acc = 0
        final_loss = 0
        total_batches = 0
        best_acc = 0
        patience = 5
        no_improve = 0
        
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            epoch_loss = 0
            correct = 0
            batches = 0
            
            for i in range(0, len(X) - batch_size, batch_size):
                batch_idx = idx[i:i+batch_size]
                batch_X = torch.FloatTensor(X[batch_idx])
                batch_y = torch.LongTensor(y[batch_idx])
                
                if np.random.random() > 0.5:
                    noise = torch.randn_like(batch_X) * 0.01
                    batch_X = batch_X + noise
                
                optimizer.zero_grad()
                outputs = finetuned_model(values=batch_X, continuous=True, task="classification")
                loss = loss_fn(outputs["base_output"], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                preds = outputs["base_output"].argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                batches += 1
            
            scheduler.step()
            final_acc = 100 * correct / len(X)
            final_loss = epoch_loss / max(batches, 1)
            total_batches += batches
            
            if final_acc > best_acc:
                best_acc = final_acc
                no_improve = 0
            else:
                no_improve += 1
            
            session["epoch"] = epoch + 1
            session["epochs"] = epochs
            session["accuracy"] = final_acc
            session["loss"] = final_loss
            session["status"] = "training"
            
            # Calculate ETA
            if session["start_time"] > 0 and epoch > 0:
                elapsed = time.time() - session["start_time"]
                time_per_epoch = elapsed / (epoch + 1)
                remaining = (epochs - epoch - 1) * time_per_epoch
                if remaining < 60:
                    session["eta"] = f"{int(remaining)}s"
                elif remaining < 3600:
                    session["eta"] = f"{int(remaining / 60)}m"
                else:
                    session["eta"] = f"{remaining / 3600:.1f}h"
            # Update legacy progress
            training_progress.update(session)
            
            print(f"Epoch {epoch+1}/{epochs}: Acc={final_acc:.1f}% (best={best_acc:.1f}%)")
            
        
        final_acc = best_acc
        session["status"] = "completed"
        session["accuracy"] = final_acc
        session["model_id"] = session.get('model_id', None)
        training_progress.update(session)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        finetuned_path = f'../checkpoints/model_finetuned_{timestamp}.pt'
        
        torch.save({
            'model_state_dict': finetuned_model.state_dict(),
            'config': finetune_config,
            'scalers': {'user': scaler},
            'encoders': {'user': le},
            'class_offsets': {'user': 0},
            'sectors': ['user'],
            'class_names': [str(c) for c in le.classes_],
            'feature_cols': feature_cols[:10],
            'norm_mean': X_mean.tolist(),
            'norm_std': X_std.tolist()
        }, finetuned_path)
        
        finetuned_models['latest'] = {
            'model': finetuned_model,
            'checkpoint': {
                'config': finetune_config,
                'scalers': {'user': scaler},
                'encoders': {'user': le},
                'class_names': [str(c) for c in le.classes_]
            },
            'path': finetuned_path
        }
        
        # Cleanup temp files
        for tf in temp_files:
            try:
                os.unlink(tf)
            except:
                pass
        
        return jsonify({
            "status": "success",
            "message": f"Fine-tuning completed with {len(dataframes)} file(s)",
            "accuracy": float(final_acc),
            "loss": float(final_loss),
            "epochs": epochs,
            "batch_size": batch_size,
            "batches": total_batches,
            "rows": len(df),
            "columns": len(feature_cols),
            "n_classes": n_classes,
            "classes": [str(c) for c in le.classes_],
            "model_path": finetuned_path,
            "model_name": f"model_finetuned_{timestamp}",
            "files_merged": len(dataframes),
            "merged_csv": merged_csv_path
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=SERVER_PORT, threads=4)
