"""
Base Model Training v9 - Self-Supervised Sector Learning
250K samples
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans

from model_base import TabularFoundationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def get_dynamic_config(n_samples, n_features, n_classes):
    complexity = (n_samples * n_classes) / 1000
    d_model = 512 if complexity > 5000 else (256 if complexity > 200 else 128)
    n_layers = 3 if n_samples > 50000 else 2
    n_latents = 128 if complexity > 1000 else (64 if complexity > 100 else 32)
    max_cols = max(64, int(np.ceil(n_features / 32) * 32))
    
    if n_samples < 500:
        batch_size, epochs, lr = 16, 50, 0.0005
    elif n_samples < 2000:
        batch_size, epochs, lr = 32, 30, 0.0003
    else:
        batch_size, epochs, lr = 64, 20, 0.0001
    
    patience = max(5, min(15, epochs // 4))
    return {'d_model': d_model, 'n_heads': 8, 'n_layers': n_layers, 'n_latents': n_latents, 
            'max_cols': max_cols, 'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'patience': patience}

def train_one_dataset(model, X, y, n_classes, n_features, device, sector_clusterer):
    n_sectors = 10
    model.update_heads(n_classes=n_classes, n_features=n_features, n_sectors=n_sectors)
    model.final_head = model.final_head.to(device)
    model.sector_head = model.sector_head.to(device)
    
    cfg = get_dynamic_config(len(X), n_features, n_classes)
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    lr = cfg['lr']
    patience = cfg['patience']
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]
    
    best_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        perm = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_X = torch.FloatTensor(X_train[batch_idx]).to(device)
            batch_y = torch.LongTensor(y_train[batch_idx]).to(device)
            
            optimizer.zero_grad()
            out = model(batch_X)
            
            # Classification loss
            logits = out['output']
            cls_loss = loss_fn(logits, batch_y)
            
            # Sector loss - global_latents clustering
            global_latents = out['global_latents'].detach().cpu().numpy().astype(np.float64)
            sector_targets = sector_clusterer.predict(global_latents)
            sector_targets = torch.LongTensor(sector_targets).to(device)
            sector_loss = loss_fn(out['sector'], sector_targets)
            
            # Update clusterer
            sector_clusterer.partial_fit(global_latents)
            
            # Total loss
            mcm_loss = out.get('mcm_loss', torch.tensor(0.0).to(device))
            midas_loss = out.get('midas_loss', torch.tensor(0.0).to(device))
            if isinstance(mcm_loss, float):
                mcm_loss = torch.tensor(mcm_loss).to(device)
            if isinstance(midas_loss, float):
                midas_loss = torch.tensor(midas_loss).to(device)
            
            loss = cls_loss + 0.5 * sector_loss + 0.1 * mcm_loss + 0.05 * midas_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(1) == batch_y).sum().item()
        
        train_acc = 100 * correct / len(X_train)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.LongTensor(y_val).to(device)
            out = model(X_val_t)
            val_acc = 100 * (out['output'].argmax(1) == y_val_t).float().mean().item()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            status = "*"
        else:
            no_improve += 1
            status = ""
        
        if (epoch + 1) % 5 == 0 or status == "*":
            print(f"    Ep {epoch+1}: train {train_acc:.1f}% | val {val_acc:.1f}% | best {best_acc:.1f}%{status}")
        
        if no_improve >= patience:
            print(f"    Patience reached at {best_acc:.1f}%")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_acc

def train(data_dir="data/base_model", max_samples=250_000):
    print("=" * 60)
    print("SchemaLabs.AI Base Model v9 - Self-Supervised Sector")
    print("250K samples for sector pattern learning")
    print("=" * 60)
    
    data_path = Path(data_dir)
    meta_files = sorted(data_path.glob("dataset_*.parquet"))
    
    total_rows = 0
    selected_meta = []
    for f in meta_files:
        df = pd.read_parquet(f)
        if total_rows + len(df) <= max_samples:
            selected_meta.append({'file': f, 'rows': len(df), 'features': len(df.columns) - 1, 'classes': df['target'].nunique()})
            total_rows += len(df)
    
    print(f"Datasets: {len(selected_meta)}, Rows: ~{total_rows:,}")
    
    config = {
        'd_model': 256, 'n_heads': 8, 'n_layers': 2, 'schema_layers': 3,
        'n_latents': 64, 'n_features': 64, 'n_classes': 10, 'n_sectors': 10,
        'n_types': 10, 'max_cols': 1024
    }
    
    model = TabularFoundationModel(config)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Sector clusterer - online learning
    sector_clusterer = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=256, n_init=3)
    sector_clusterer.fit(np.random.randn(256, 256))  # Initialize
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Resume from checkpoint
    start_idx = 0
    accuracies = []
    checkpoint_file = checkpoint_dir / "base_model_v9_checkpoint.pt"
    if checkpoint_file.exists():
        print(f"Loading checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        start_idx = ckpt.get('datasets_trained', 0)
        accuracies = ckpt.get('accuracies', [])
        if 'sector_centers' in ckpt:
            sector_clusterer.cluster_centers_ = ckpt['sector_centers']
        print(f"Resuming from dataset {start_idx}, avg accuracy: {ckpt.get('avg_accuracy', 0):.1f}%")
    
    start_time = datetime.now()
    
    for i, meta in enumerate(selected_meta):
        if i < start_idx:
            continue
        df = pd.read_parquet(meta['file'])
        X = df.drop('target', axis=1).values.astype(np.float32)
        y = df['target'].values
        
        X = np.nan_to_num(X, nan=0.0)
        mean, std = X.mean(0), X.std(0) + 1e-8
        X = (X - mean) / std
        
        n_classes = int(df['target'].nunique())
        n_features = X.shape[1]
        
        print(f"\n[{i+1}/{len(selected_meta)}] {meta['file'].name} - {n_classes} cls, {n_features} feat, {len(X)} rows")
        
        val_acc = train_one_dataset(model, X, y, n_classes, n_features, device, sector_clusterer)
        accuracies.append(val_acc)
        
        print(f"  => Val: {val_acc:.1f}% | Avg: {np.mean(accuracies):.1f}%")
        
        if (i + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'datasets_trained': i + 1,
                'avg_accuracy': np.mean(accuracies),
                'accuracies': accuracies,
                'sector_centers': sector_clusterer.cluster_centers_
            }, checkpoint_dir / "base_model_v9_checkpoint.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'avg_accuracy': np.mean(accuracies),
        'accuracies': accuracies,
        'sector_centers': sector_clusterer.cluster_centers_
    }, checkpoint_dir / "base_model_v9_final.pt")
    
    print("\n" + "=" * 60)
    print(f"Done! Avg Val Accuracy: {np.mean(accuracies):.1f}%")
    print(f"Time: {datetime.now() - start_time}")
    print("=" * 60)

if __name__ == "__main__":
    train()
