"""
SchemaLabs.AI Base Model Training - GPU
83 Component - Full 10M Data in RAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from datetime import datetime
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class CellProcessing(nn.Module):
    def __init__(self, max_features, d_model):
        super().__init__()
        self.value_embed = nn.Linear(1, d_model)
        self.col_embed = nn.Embedding(max_features, d_model)
        self.pos_embed = nn.Embedding(max_features, d_model)
        self.type_embed = nn.Embedding(4, d_model)
        self.fusion = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        batch, n_feat = x.shape
        val_emb = self.value_embed(x.unsqueeze(-1))
        col_ids = torch.arange(n_feat, device=x.device)
        col_emb = self.col_embed(col_ids).unsqueeze(0).expand(batch, -1, -1)
        pos_emb = self.pos_embed(col_ids).unsqueeze(0).expand(batch, -1, -1)
        type_ids = torch.zeros(n_feat, dtype=torch.long, device=x.device)
        type_emb = self.type_embed(type_ids).unsqueeze(0).expand(batch, -1, -1)
        return self.norm(self.fusion(torch.cat([val_emb, col_emb, pos_emb, type_emb], dim=-1)))


class SchemaProcessing(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout=0.1, batch_first=True) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.proj(x.mean(dim=1))


class LocalReasoning(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(0.1))
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        row_out, _ = self.row_attn(x, x, x)
        x = self.norm1(x + row_out)
        return self.norm2(x + self.ffn(x))


class GlobalReasoning(nn.Module):
    def __init__(self, d_model, n_heads, n_latents=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
    def forward(self, x):
        batch = x.size(0)
        latents = self.latents.expand(batch, -1, -1)
        latents, _ = self.cross_attn(latents, x, x)
        latents = self.norm1(latents)
        self_out, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + self_out)
        return latents, (latents + self.ffn(latents)).mean(dim=1)


class MIDAS(nn.Module):
    def __init__(self, max_features, d_hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(max_features*2, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden//2))
        self.decoder = nn.Sequential(nn.Linear(d_hidden//2, d_hidden), nn.ReLU(), nn.Linear(d_hidden, max_features))
    def forward(self, x, mask=None):
        if mask is None: mask = torch.ones_like(x)
        z = self.encoder(torch.cat([x*mask, mask], dim=-1))
        recon = self.decoder(z)
        loss = F.mse_loss(recon*(1-mask), x*(1-mask), reduction='sum') / ((1-mask).sum()+1e-8) if self.training and (1-mask).sum()>0 else torch.tensor(0.0, device=x.device)
        return x*mask + recon*(1-mask), loss


class MCMHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, cell_grid, x_orig):
        batch, n_feat, d = cell_grid.shape
        mask = torch.rand(batch, n_feat, device=cell_grid.device) < 0.15
        pred = self.proj(cell_grid).squeeze(-1)
        loss = F.mse_loss(pred[mask], x_orig[:,:n_feat][mask]) if mask.any() and self.training else torch.tensor(0.0, device=cell_grid.device)
        return pred, loss


class TimeSeriesModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trend = nn.Linear(d_model, 1)
        self.seasonal = nn.Linear(d_model, 4)
    def forward(self, x):
        return {"trend": self.trend(x.mean(1)), "seasonal": self.seasonal(x.mean(1))}


class AnalyticsEngine(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.stats = nn.Linear(d_model, 8)
        self.outlier = nn.Linear(d_model, 2)
        self.cluster = nn.Linear(d_model, 16)
    def forward(self, x):
        return {"stats": self.stats(x), "outlier": self.outlier(x), "cluster": self.cluster(x)}


class MIRASModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.retention_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.low_rank_down = nn.Linear(d_model, 32)
        self.low_rank_up = nn.Linear(32, d_model)
        self.swiglu_w1 = nn.Linear(d_model, d_model * 2)
        self.swiglu_w2 = nn.Linear(d_model, d_model * 2)
        self.swiglu_w3 = nn.Linear(d_model * 2, d_model)
    def forward(self, x):
        x = x * self.retention_gate(x)
        x = x + self.low_rank_up(self.low_rank_down(x)) * 0.1
        x = x + self.swiglu_w3(F.silu(self.swiglu_w1(x)) * self.swiglu_w2(x)) * 0.1
        return x


class SectorHead(nn.Module):
    def __init__(self, d_model, n_sectors=50):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, n_sectors))
    def forward(self, x):
        return self.head(x)


class BaseModel83(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_features = config["max_features"]
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        n_latents = config["n_latents"]
        n_sectors = config["n_sectors"]
        max_classes = config["max_classes"]
        
        self.max_features = max_features
        self.d_model = d_model
        
        self.cell_processing = CellProcessing(max_features, d_model)
        self.schema_processing = SchemaProcessing(d_model, n_heads, n_layers)
        self.local_reasoning = LocalReasoning(d_model, n_heads)
        self.global_reasoning = GlobalReasoning(d_model, n_heads, n_latents)
        self.midas = MIDAS(max_features, d_model)
        self.mcm_head = MCMHead(d_model)
        self.time_series = TimeSeriesModule(d_model)
        self.analytics = AnalyticsEngine(d_model)
        self.miras = MIRASModule(d_model)
        self.sector_head = SectorHead(d_model, n_sectors)
        
        self.input_proj = nn.Sequential(nn.Linear(max_features, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.final_fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(d_model, max_classes)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x, n_classes=10):
        x = torch.clamp(x, -5.0, 5.0)
        mask = torch.ones_like(x)
        x_imp, midas_loss = self.midas(x, mask)
        cell = self.cell_processing(x_imp)
        schema = self.schema_processing(cell)
        local = self.local_reasoning(cell)
        latents, glob = self.global_reasoning(local)
        ts = self.time_series(local)
        analytics = self.analytics(glob)
        miras = self.miras(glob)
        _, mcm_loss = self.mcm_head(cell, x_imp)
        inp = self.input_proj(x_imp)
        combined = torch.cat([glob, miras, inp], dim=-1)
        features = self.final_fusion(combined)
        logits = self.classifier(features)[:, :n_classes]
        return {"logits": logits, "features": features, "midas_loss": midas_loss, "mcm_loss": mcm_loss}


def load_all_data(data_dir, max_features):
    """Tüm data'yı RAM'e yükle"""
    data_dir = Path(data_dir)
    
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    all_X, all_y, all_cls = [], [], []
    
    print(f"Loading {len(metadata)} datasets into RAM...")
    for meta in tqdm(metadata):
        df = pd.read_parquet(data_dir / meta["filename"])
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.int64)
        
        X = np.nan_to_num(X, nan=0.0)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        if X.shape[1] < max_features:
            X = np.hstack([X, np.zeros((X.shape[0], max_features - X.shape[1]), dtype=np.float32)])
        elif X.shape[1] > max_features:
            X = X[:, :max_features]
        
        all_X.append(X)
        all_y.append(y)
        all_cls.extend([meta["n_classes"]] * len(y))
    
    X = np.vstack(all_X)
    y = np.hstack(all_y)
    n_cls = np.array(all_cls)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y, n_cls = X[idx], y[idx], n_cls[idx]
    
    print(f"Loaded: {len(X):,} samples, {X.shape[1]} features")
    print(f"Memory: {X.nbytes / 1e9:.2f} GB")
    
    return X, y, n_cls, metadata


def train(data_dir="data/base_model"):
    print("=" * 60)
    print("SchemaLabs.AI Base Model - GPU (Full RAM)")
    print("=" * 60)
    
    with open(Path(data_dir) / "metadata.json") as f:
        metadata = json.load(f)
    
    max_features = max(m["n_features"] for m in metadata) + 10
    max_classes = max(m["n_classes"] for m in metadata) + 10
    
    print(f"Max features: {max_features}, Max classes: {max_classes}")
    
    # Load all data
    X, y, n_cls, metadata = load_all_data(data_dir, max_features)
    
    # Split
    n_train = int(len(X) * 0.9)
    X_train, y_train, cls_train = X[:n_train], y[:n_train], n_cls[:n_train]
    X_val, y_val, cls_val = X[n_train:], y[n_train:], n_cls[n_train:]
    
    del X, y, n_cls
    gc.collect()
    
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    # Config
    batch_size = 512
    epochs = 50
    lr = 0.001
    
    print(f"Config: batch={batch_size}, epochs={epochs}, lr={lr}")
    
    # DataLoader - max_classes for all
    train_loader = DataLoader(
        list(zip(X_train, y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: (torch.FloatTensor(np.array([x[0] for x in b])), torch.LongTensor([x[1] for x in b]))
    )
    val_loader = DataLoader(
        list(zip(X_val, y_val)),
        batch_size=batch_size*2, shuffle=False, num_workers=0,
        collate_fn=lambda b: (torch.FloatTensor(np.array([x[0] for x in b])), torch.LongTensor([x[1] for x in b]))
    )
    
    print(f"Batches: {len(train_loader)} train, {len(val_loader)} val")
    
    config = {"max_features": max_features, "max_classes": max_classes, "d_model": 256, "n_heads": 8, "n_layers": 3, "n_latents": 64, "n_sectors": 50}
    
    model = BaseModel83(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_val_acc = 0
    patience = 10
    no_improve = 0
    
    start_time = datetime.now()
    
    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    out = model(X_batch, n_classes=max_classes)
                    loss = F.cross_entropy(out["logits"], y_batch) + out["midas_loss"]*0.1 + out["mcm_loss"]*0.1
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(X_batch, n_classes=max_classes)
                loss = F.cross_entropy(out["logits"], y_batch) + out["midas_loss"]*0.1 + out["mcm_loss"]*0.1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            pred = out["logits"].argmax(1)
            train_correct += (pred == y_batch).sum().item()
            train_total += y_batch.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*train_correct/train_total:.1f}%"})
        
        scheduler.step()
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch, n_classes=max_classes)
                val_correct += (out["logits"].argmax(1) == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config, "accuracy": best_val_acc}, checkpoint_dir / "base_model_best.pt")
            status = " ★"
        else:
            no_improve += 1
        
        print(f"  Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%{status}")
        
        if best_val_acc >= 95.0 or no_improve >= patience:
            break
    
    print(f"\nDone! Best Val: {best_val_acc:.1f}% | Time: {datetime.now()-start_time}")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "accuracy": best_val_acc}, checkpoint_dir / "base_model_final.pt")


if __name__ == "__main__":
    train(data_dir="data/base_model")
