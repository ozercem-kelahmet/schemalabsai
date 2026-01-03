import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value)

MODEL_PATH = os.getenv('MODEL_PATH')

# MIDAS - Base model'den gelecek
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
    def encode(self, x):
        """Feature extraction - mask=1 (tüm data var)"""
        mask = torch.ones_like(x)
        return self.encoder(torch.cat([x, mask], dim=1))

class FineTuneHead(nn.Module):
    """Sadece classification head - base encoder frozen"""
    def __init__(self, input_dim=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class FineTuneModel(nn.Module):
    """Backward compatibility için - standalone model"""
    def __init__(self, n_features, n_classes):
        super().__init__()
        hidden = min(512, max(128, n_features * 4))
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, n_classes)
        )
    def forward(self, x): 
        return self.net(x)

def analyze_data_quality(X, y, classes):
    """Data kalitesini analiz et ve realistic target belirle"""
    n_samples = len(X)
    n_features = X.shape[1]
    n_classes = len(classes)
    missing_pct = 100 * np.isnan(X).sum() / X.size if X.size > 0 else 0
    
    class_counts = np.bincount(y.astype(int), minlength=n_classes)
    class_balance = class_counts.min() / max(class_counts.max(), 1)
    
    X_filled = np.nan_to_num(X, nan=0)
    feature_vars = np.var(X_filled, axis=0)
    useful_features = int((feature_vars > 0.01).sum())
    
    # Base model transfer ile daha yüksek target
    base_target = 0.95
    if missing_pct > 50: base_target -= 0.10
    elif missing_pct > 30: base_target -= 0.05
    
    if class_balance < 0.3: base_target -= 0.05
    
    if n_classes > 50: base_target -= 0.10
    elif n_classes > 20: base_target -= 0.05
    
    return {
        'n_samples': int(n_samples),
        'n_features': int(n_features),
        'n_classes': int(n_classes),
        'missing_pct': float(missing_pct),
        'class_balance': float(class_balance),
        'useful_features': useful_features,
        'realistic_target': float(max(0.60, min(0.99, base_target)))
    }

def merge_dataframes(dfs):
    if len(dfs) == 1:
        return dfs[0]
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    return pd.concat([d[list(common_cols)] for d in dfs], ignore_index=True)

def clean_data(df, target_column):
    if target_column in df.columns:
        df = df[df[target_column].notna()].copy()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    valid_cols = [c for c in numeric_cols if df[c].notna().sum() > len(df) * 0.1]
    return df, valid_cols

def encode_target(y):
    if y.dtype == 'object' or y.dtype.name == 'category':
        classes = [c for c in y.unique() if pd.notna(c)]
        class_to_id = {c: i for i, c in enumerate(classes)}
        id_to_class = {i: c for c, i in class_to_id.items()}
        y_encoded = np.array([class_to_id.get(v, -1) for v in y])
        valid_mask = y_encoded >= 0
        return y_encoded, classes, id_to_class, valid_mask
    else:
        y = y.fillna(0).values
        unique = np.unique(y[~np.isnan(y)])
        if len(unique) > 50:
            y = pd.qcut(y, q=min(20, len(unique)), labels=False, duplicates='drop')
        classes = [str(i) for i in range(int(np.nanmax(y)) + 1)]
        id_to_class = {i: c for i, c in enumerate(classes)}
        return y.astype(np.int64), classes, id_to_class, np.ones(len(y), dtype=bool)

def finetune_model(
    filepaths,
    target_column=None,
    max_epochs=200,
    batch_size=64,
    learning_rate=0.001,
    patience=30,
    midas=None,
    output_path=None,
    progress_callback=None
):
    """
    Transfer Learning Fine-tune:
    1. Base model'in MIDAS encoder'ını kullan (frozen)
    2. Sadece classification head'i eğit
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    # Load files
    dfs = []
    for fp in filepaths:
        try:
            if fp.endswith('.csv'):
                dfs.append(pd.read_csv(fp))
            elif fp.endswith(('.xlsx', '.xls')):
                dfs.append(pd.read_excel(fp))
            elif fp.endswith('.parquet'):
                dfs.append(pd.read_parquet(fp))
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    
    if not dfs:
        raise ValueError("No valid files")
    
    df = merge_dataframes(dfs)
    
    # Auto-detect target
    if not target_column:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        target_column = cat_cols[0] if cat_cols else df.columns[-1]
    
    # Clean
    df, feature_cols = clean_data(df, target_column)
    if not feature_cols:
        raise ValueError("No valid features")
    
    # Encode target
    y, classes, id_to_class, valid_mask = encode_target(df[target_column])
    df = df[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    
    X = df[feature_cols].values.astype(np.float32)
    n_features_orig = len(feature_cols)
    n_classes = len(classes)
    
    # Analyze quality
    quality = analyze_data_quality(X, y, classes)
    target_accuracy = quality['realistic_target']
    
    # ========== TRANSFER LEARNING ==========
    # Load base model MIDAS if not provided
    if midas is None and MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f"Loading base model from {MODEL_PATH}...")
        ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        midas = MIDAS(10, 512)
        midas.load_state_dict(ckpt['midas'])
    
    use_transfer = midas is not None
    
    if use_transfer:
        print("Using Transfer Learning with base model encoder")
        midas.eval()
        # Freeze encoder
        for p in midas.parameters():
            p.requires_grad = False
        
        # Prepare data for MIDAS (10 features)
        # Normalize first
        X_min, X_max = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        
        # Pad/trim to 10 features
        if X_norm.shape[1] < 10:
            X_pad = np.hstack([X_norm, np.zeros((X_norm.shape[0], 10 - X_norm.shape[1]), dtype=np.float32)])
        else:
            X_pad = X_norm[:, :10]
        
        # Impute missing values
        X_tensor = torch.FloatTensor(X_pad)
        mask = torch.ones_like(X_tensor)  # Assuming no missing after fillna
        
        # Extract features with MIDAS encoder
        print("Extracting features with base model encoder...")
        with torch.no_grad():
            features_list = []
            for i in range(0, len(X_tensor), 1000):
                batch = X_tensor[i:i+1000]
                feat = midas.encode(batch)  # 256 dim
                features_list.append(feat)
            X_features = torch.cat(features_list)
        
        print(f"Extracted features shape: {X_features.shape}")
        
        # Use extracted features for training
        n_features = 256  # MIDAS encoder output
        X_train_data = X_features
        
        # Classification head only
        model = FineTuneHead(input_dim=256, n_classes=n_classes)
    else:
        print("No base model - training from scratch")
        # Fallback to standalone model
        X_norm = np.nan_to_num(X, nan=0.0)
        col_means = np.nanmean(X_norm, axis=0)
        for i in range(X_norm.shape[1]):
            mask = X_norm[:, i] == 0
            if col_means[i] != 0:
                X_norm[mask, i] = col_means[i]
        
        X_min, X_max = X_norm.min(0), X_norm.max(0)
        X_norm = (X_norm - X_min) / (X_max - X_min + 1e-8)
        
        X_train_data = torch.FloatTensor(X_norm)
        n_features = n_features_orig
        model = FineTuneModel(n_features, n_classes)
    
    # Split
    perm = np.random.permutation(len(X_train_data))
    X_train_data = X_train_data[perm]
    y = y[perm]
    split = int(len(X_train_data) * 0.8)
    
    X_train = X_train_data[:split]
    y_train = torch.LongTensor(y[:split])
    X_test = X_train_data[split:]
    y_test = torch.LongTensor(y[split:])
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {n_classes}")
    
    # Training
    opt = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    best_acc, best_state, patience_counter, final_epoch = 0, None, 0, 0
    
    for ep in range(max_epochs):
        model.train()
        total_loss = 0
        for bi, (bx, by) in enumerate(loader):
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if progress_callback:
                progress_callback({
                    'status': 'training',
                    'epoch': ep + 1,
                    'epochs': max_epochs,
                    'batch': bi + 1,
                    'batches': len(loader),
                    'loss': total_loss / (bi + 1),
                    'accuracy': best_acc
                })
        
        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(-1) == y_test).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  Epoch {ep+1}: {acc*100:.1f}% ✓")
        else:
            patience_counter += 1
        
        final_epoch = ep + 1
        
        if best_acc >= target_accuracy:
            print(f"  🎉 Reached target {target_accuracy*100:.0f}% at epoch {ep+1}")
            break
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {ep+1}")
            break
    
    model.load_state_dict(best_state)
    
    # Per-class accuracy
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(-1)
    
    class_accs = {}
    for i, cls in enumerate(classes):
        mask = y_test == i
        if mask.sum() > 0:
            class_accs[str(cls)] = float((pred[mask] == y_test[mask]).float().mean().item())
    
    # Save
    if output_path is None:
        output_path = str(Path(filepaths[0]).with_suffix('.finetuned.pt'))
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    save_dict = {
        'model': model.state_dict(),
        'n_features': n_features,
        'n_classes': n_classes,
        'feature_cols': feature_cols,
        'classes': classes,
        'id_to_class': id_to_class,
        'accuracy': best_acc,
        'class_accuracies': class_accs,
        'quality': quality,
        'epochs_trained': final_epoch,
        'target_column': target_column,
        'use_transfer': use_transfer
    }
    
    if not use_transfer:
        save_dict['X_min'] = X_min.tolist()
        save_dict['X_max'] = X_max.tolist()
    
    torch.save(save_dict, output_path)
    
    return {
        'status': 'complete',
        'accuracy': best_acc,
        'model_path': output_path,
        'n_classes': n_classes,
        'n_features': n_features,
        'epochs': final_epoch,
        'classes': classes,
        'quality': quality,
        'loss': total_loss / len(loader),
        'use_transfer': use_transfer
    }

def analyze_only(filepaths, target_column=None):
    """Sadece analiz yap"""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    dfs = []
    for fp in filepaths:
        try:
            if fp.endswith('.csv'):
                dfs.append(pd.read_csv(fp))
            elif fp.endswith(('.xlsx', '.xls')):
                dfs.append(pd.read_excel(fp))
        except:
            pass
    
    if not dfs:
        return {'error': 'No valid files'}
    
    df = merge_dataframes(dfs)
    
    if not target_column:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        target_column = cat_cols[0] if cat_cols else df.columns[-1]
    
    df, feature_cols = clean_data(df, target_column)
    y, classes, _, valid_mask = encode_target(df[target_column])
    X = df[feature_cols].values.astype(np.float32)
    y = y[valid_mask]
    
    quality = analyze_data_quality(X, y, classes)
    
    return {
        'status': 'analyzed',
        'quality': quality,
        'target_column': target_column,
        'feature_columns': feature_cols,
        'classes': classes[:50],
        'n_rows': len(df)
    }

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python finetune.py <file> [--target col] [--epochs N]")
        sys.exit(1)
    
    files, target, epochs = [], None, 200
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--target':
            target = sys.argv[i+1]; i += 2
        elif sys.argv[i] == '--epochs':
            epochs = int(sys.argv[i+1]); i += 2
        else:
            files.append(sys.argv[i]); i += 1
    
    print("=" * 60)
    print("SCHEMALABS AI - Transfer Learning Fine-tune")
    print("=" * 60)
    
    result = finetune_model(files, target, epochs)
    print(f"\n✅ Accuracy: {result['accuracy']*100:.1f}%")
    print(f"   Transfer: {result['use_transfer']}")
    print(f"   Model: {result['model_path']}")
