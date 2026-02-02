"""
SchemaLabs.AI Base Model v0_1M
TabularFoundationModel - 83 component
Fine-tune ile BİREBİR AYNI training loop
TEK model, TÜM dataset'lerden öğrenir
%99 accuracy hedef
"""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '/opt/schemalabsai/model')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from datetime import datetime

from model_base import TabularFoundationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def get_dynamic_config(n_samples, n_features, n_classes):
    """Fine-tune ile AYNI dynamic config"""
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
    
    # %99 hedef için daha fazla epoch
    if n_samples < 100:
        epochs = 150
    elif n_samples < 500:
        epochs = 100
    elif n_samples < 2000:
        epochs = 60
    else:
        epochs = 40
    
    if n_classes > 20:
        epochs = min(epochs + 20, 200)
    
    patience = max(5, min(15, epochs // 4))  # Daha uzun patience
    
    if n_samples < 500:
        lr = 0.0005
    elif n_samples < 2000:
        lr = 0.0003
    elif batch_size <= 8:
        lr = 0.0001
    elif batch_size <= 32:
        lr = 0.0002
    else:
        lr = 0.0001
    
    return {
        'd_model': d_model,
        'n_heads': 8,
        'n_layers': n_layers,
        'n_latents': n_latents,
        'max_cols': max_cols,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'patience': patience
    }


def load_dataset(filepath):
    df = pd.read_parquet(filepath)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    X = np.nan_to_num(X, nan=0.0)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    return X, y


def train_one_dataset(model, X, y, n_classes, n_features, device, backbone_optimizer, scaler, train_backbone=True):
    """Fine-tune ile BİREBİR AYNI training loop - %99 hedef"""
    
    
    # Head'leri güncelle
    n_sectors = min(n_classes, 10)
    model.update_heads(n_classes=n_classes, n_features=n_features, n_sectors=n_sectors)
    model.final_head = model.final_head.to(device)
    model.sector_head = model.sector_head.to(device)
    
    # Dynamic config
    cfg = get_dynamic_config(len(X), n_features, n_classes)
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    lr = cfg['lr']
    patience = cfg['patience']
    
    # Optimizer
    if train_backbone:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        head_params = list(model.final_head.parameters()) + list(model.sector_head.parameters())
        optimizer = AdamW(head_params, lr=lr, weight_decay=0.01)
    
    # Warmup + Cosine scheduler
    warmup_epochs = min(3, epochs // 5)
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    loss_fn = nn.CrossEntropyLoss()
    gradient_accumulation_steps = 2
    
    if len(X) > 10000:
        max_samples_per_epoch = 10000
    else:
        max_samples_per_epoch = len(X)
    
    # Train/Val split
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.9)
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]
    
    # DataLoader
    dataset = TensorDataset(torch.FloatTensor(X_train[:max_samples_per_epoch]), torch.LongTensor(y_train[:max_samples_per_epoch]))
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_mem = torch.cuda.is_available()
    prefetch = 2 if num_workers > 0 else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, prefetch_factor=prefetch)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    best_acc = 0
    best_state = None
    no_improve = 0
    max_epochs = 500
    current_epoch = 0
    
    while current_epoch < max_epochs:
        model.train()
        total_loss = 0
        correct = 0
        batches = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            if np.random.random() > 0.5:
                noise = torch.randn_like(batch_X) * 0.01
                batch_X = batch_X + noise
                batch_X = batch_X.contiguous()
            
            with autocast('cuda'):
                out = model(batch_X)
                logits = out['output']
                sector_logits = out['sector']
                
                sector_targets = batch_y % model.n_sectors
                sector_loss = nn.CrossEntropyLoss()(sector_logits, sector_targets)
                
                mcm_loss = out.get('mcm_loss', torch.tensor(0.0, device=device))
                miras_loss = out.get('miras_loss', torch.tensor(0.0, device=device))
                
                loss = loss_fn(logits, batch_y) + sector_loss + 0.1 * mcm_loss + 0.05 * miras_loss
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # NaN/Inf check
                valid_gradients = True
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break
                if valid_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            correct += (logits.argmax(1) == batch_y).sum().item()
            batches += 1
        
        current_epoch += 1
        scheduler.step()
        acc = 100 * correct / min(len(X_train), max_samples_per_epoch)
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(X_val_t)
            val_pred = out['output'].argmax(1)
            val_acc = 100 * (val_pred == y_val_t).sum().item() / len(y_val_t)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            status = "★"
        else:
            no_improve += 1
            status = ""
        
        bb = "BB" if train_backbone else "FZ"
        print(f"    Ep {current_epoch} [{bb}]: train {acc:.1f}% | val {val_acc:.1f}% | best {best_acc:.1f}%{status}")
        
        # Early stop - SADECE %99'da dur
        if best_acc >= 99.0:
            print(f"    🎉 %99+ reached!")
            break
        
        # Çok uzun süre improvement yoksa dur
        if no_improve >= patience:
            print(f"    ⏹ Patience reached at {best_acc:.1f}%")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_acc


def train(data_dir="data/base_model", max_samples=1_000_000):
    print("=" * 60)
    print("SchemaLabs.AI Base Model v0_1M")
    print("TabularFoundationModel - 83 component")
    print("Fine-tune ile BİREBİR AYNI")
    print("TEK model, TÜM dataset'lerden öğrenir")
    print("%99 accuracy hedef")
    print("=" * 60)
    
    with open(Path(data_dir) / "metadata.json") as f:
        metadata = json.load(f)
    
    # Feature agnostic - max_features gerekmez
    
    total_rows = 0
    selected_meta = []
    for m in metadata:
        if total_rows >= max_samples:
            break
        selected_meta.append(m)
        total_rows += m["n_rows"]
    
    print(f"Datasets: {len(selected_meta)}, Rows: ~{total_rows:,}")
    print(f"Feature/Class AGNOSTIC")
    
    config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 3,
        'schema_layers': 3,
        'n_latents': 64,
        'n_features': 64,  # başlangıç, agnostic
        'n_classes': 10,
        'n_sectors': 10,
        'n_types': 10,
        'max_cols': 1024  # yeterince büyük
    }
    
    model = TabularFoundationModel(config)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    backbone_params = []
    for name, param in model.named_parameters():
        if 'final_head' not in name and 'sector_head' not in name and 'values_proj' not in name:
            backbone_params.append(param)
    backbone_optimizer = AdamW(backbone_params, lr=0.001, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Resume from checkpoint
    start_idx = 0
    accuracies = []
    checkpoint_file = checkpoint_dir / "base_model_v0_1M_checkpoint.pt"
    if checkpoint_file.exists():
        print(f"Loading checkpoint: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location=device)
        # Sadece backbone weights yükle, head'leri atla
        state_dict = ckpt['model_state_dict']
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered)}/{len(state_dict)} layers")
        start_idx = ckpt.get('datasets_trained', 0)
        accuracies = ckpt.get('accuracies', [])
        print(f"Resuming from dataset {start_idx}, avg accuracy: {ckpt.get('avg_accuracy', 0):.1f}%")
    
    start_time = datetime.now()
    
    backbone_train_count = 96
    
    for i, meta in enumerate(selected_meta):
        if i < start_idx:
            continue
        train_backbone = (i < backbone_train_count)
        mode = "BACKBONE" if train_backbone else "FROZEN"
        
        print(f"\n[{i+1}/{len(selected_meta)}] [{mode}] {meta['filename']} - {meta['n_classes']} cls, {meta['n_features']} feat, {meta['n_rows']} rows")
        
        X, y = load_dataset(Path(data_dir) / meta["filename"])
        n_classes = meta["n_classes"]
        n_features = meta["n_features"]
        
        val_acc = train_one_dataset(model, X, y, n_classes, n_features, device, backbone_optimizer, scaler, train_backbone)
        accuracies.append(val_acc)
        
        avg_acc = sum(accuracies) / len(accuracies)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"  => Val: {val_acc:.1f}% | Avg: {avg_acc:.1f}% | Time: {elapsed:.0f}s")
        
        if i == backbone_train_count - 1:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "avg_accuracy": avg_acc,
                "datasets_trained": i + 1
            }, checkpoint_dir / "base_model_v0_1M_backbone.pt")
            print(f"\n*** Backbone training complete! ***\n")
        
        if (i + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "avg_accuracy": avg_acc,
                "datasets_trained": i + 1
            }, checkpoint_dir / "base_model_v0_1M_checkpoint.pt")
    
    final_avg = sum(accuracies) / len(accuracies)
    print(f"\n{'='*60}")
    print(f"Done! Avg Val Accuracy: {final_avg:.1f}%")
    print(f"Time: {datetime.now()-start_time}")
    print(f"{'='*60}")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "avg_accuracy": final_avg,
        "datasets_trained": len(selected_meta),
        "accuracies": accuracies
    }, checkpoint_dir / "base_model_v0_1M_final.pt")


if __name__ == "__main__":
    train(data_dir="data/base_model", max_samples=1_000_000)
