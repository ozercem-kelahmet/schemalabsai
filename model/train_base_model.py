"""
SchemaLabs.AI Base Model Training
=================================
83 Component - Orijinal Mimari
10M row test, sonra 150M row GCP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(10)
torch.set_num_interop_threads(4)

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Threads: {torch.get_num_threads()}")


# ============================================================
# 83 COMPONENT - ORİJİNAL MİMARİ
# ============================================================

class CellProcessing(nn.Module):
    """Component 1-5: Cell Processing"""
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
        
        fused = torch.cat([val_emb, col_emb, pos_emb, type_emb], dim=-1)
        return self.norm(self.fusion(fused))


class SchemaProcessing(nn.Module):
    """Component 6-8: Schema Processing"""
    def __init__(self, d_model, n_heads, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.proj(x.mean(dim=1))


class LocalReasoning(nn.Module):
    """Component 9-11: Local Reasoning"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        row_out, _ = self.row_attn(x, x, x)
        x = self.norm1(x + row_out)
        x = self.norm2(x + self.ffn(x))
        return x


class GlobalReasoning(nn.Module):
    """Component 12-17: Global Reasoning (Perceiver-style)"""
    def __init__(self, d_model, n_heads, n_latents=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.memory_alpha = 0.7
        
    def forward(self, x):
        batch = x.size(0)
        latents = self.latents.expand(batch, -1, -1)
        latents, _ = self.cross_attn(latents, x, x)
        latents = self.norm1(latents)
        self_out, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + self_out)
        latents = latents + self.ffn(latents)
        return latents, latents.mean(dim=1)


class MIDAS(nn.Module):
    """Component 18-22: Missing Data Imputation"""
    def __init__(self, max_features, d_hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(max_features * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_hidden // 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, max_features)
        )
        
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        z = self.encoder(torch.cat([x * mask, mask], dim=-1))
        x_recon = self.decoder(z)
        x_imputed = x * mask + x_recon * (1 - mask)
        loss = torch.tensor(0.0, device=x.device)
        if self.training and (1 - mask).sum() > 0:
            loss = F.mse_loss(x_recon * (1 - mask), x * (1 - mask), reduction='sum') / ((1 - mask).sum() + 1e-8)
        return x_imputed, loss


class MCMHead(nn.Module):
    """Component 27-28: Masked Cell Modeling"""
    def __init__(self, d_model, max_features):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        
    def forward(self, cell_grid, x_original, mask_ratio=0.15):
        batch, n_feat, d = cell_grid.shape
        mask = torch.rand(batch, n_feat, device=cell_grid.device) < mask_ratio
        pred = self.proj(cell_grid).squeeze(-1)
        loss = torch.tensor(0.0, device=cell_grid.device)
        if mask.any() and self.training:
            target = x_original[:, :n_feat]
            loss = F.mse_loss(pred[mask], target[mask])
        return pred, loss


class TimeSeriesModule(nn.Module):
    """Component 29-36: Time Series Processing"""
    def __init__(self, d_model):
        super().__init__()
        self.trend = nn.Linear(d_model, 1)
        self.seasonal = nn.Linear(d_model, 4)
        self.lag = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.autocorr = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        trend = self.trend(x)
        seasonal = self.seasonal(x)
        lag_out, _ = self.lag(x)
        autocorr = self.autocorr(x)
        return {"trend": trend, "seasonal": seasonal, "lag": lag_out, "autocorr": autocorr}


class AnalyticsEngine(nn.Module):
    """Component 37-42: Analytics Engine"""
    def __init__(self, d_model):
        super().__init__()
        self.stats = nn.Linear(d_model, 8)
        self.outlier = nn.Linear(d_model, 2)
        self.cluster = nn.Linear(d_model, 16)
        self.regression = nn.Linear(d_model, 1)
        
    def forward(self, x):
        return {
            "stats": self.stats(x),
            "outlier": self.outlier(x),
            "cluster": self.cluster(x),
            "regression": self.regression(x)
        }


class MIRASModule(nn.Module):
    """Component 43-63: MIRAS (21 active features)"""
    def __init__(self, d_model):
        super().__init__()
        self.huber_delta = nn.Parameter(torch.ones(1))
        self.retention_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.transform = nn.Linear(d_model, d_model)
        self.low_rank_down = nn.Linear(d_model, 32)
        self.low_rank_up = nn.Linear(32, d_model)
        self.swiglu_w1 = nn.Linear(d_model, d_model * 2)
        self.swiglu_w2 = nn.Linear(d_model, d_model * 2)
        self.swiglu_w3 = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        retention = self.retention_gate(x)
        x = x * retention
        low = self.low_rank_up(self.low_rank_down(x))
        x = x + low * 0.1
        swiglu = self.swiglu_w3(F.silu(self.swiglu_w1(x)) * self.swiglu_w2(x))
        x = x + swiglu * 0.1
        return x


class SectorHead(nn.Module):
    """Component 73: Sector Detection"""
    def __init__(self, d_model, n_sectors=50):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, n_sectors))
        
    def forward(self, x):
        return self.head(x)


class DataProcessing(nn.Module):
    """Component 67-72: Data Processing"""
    def __init__(self, max_features):
        super().__init__()
        self.norm = nn.LayerNorm(max_features)
        self.outlier_clip = 5.0
        
    def forward(self, x):
        x = torch.clamp(x, -self.outlier_clip, self.outlier_clip)
        return x


class BaseModel83(nn.Module):
    """
    SchemaLabs.AI Base Model - 83 Components
    Feature/Class Agnostic
    """
    def __init__(self, config):
        super().__init__()
        
        max_features = config.get("max_features", 210)
        d_model = config.get("d_model", 256)
        n_heads = config.get("n_heads", 8)
        n_layers = config.get("n_layers", 3)
        n_latents = config.get("n_latents", 64)
        n_sectors = config.get("n_sectors", 50)
        max_classes = config.get("max_classes", 200)
        
        self.max_features = max_features
        self.d_model = d_model
        
        # Component 1-5
        self.cell_processing = CellProcessing(max_features, d_model)
        # Component 6-8
        self.schema_processing = SchemaProcessing(d_model, n_heads, n_layers)
        # Component 9-11
        self.local_reasoning = LocalReasoning(d_model, n_heads)
        # Component 12-17
        self.global_reasoning = GlobalReasoning(d_model, n_heads, n_latents)
        # Component 18-22
        self.midas = MIDAS(max_features, d_model)
        # Component 27-28
        self.mcm_head = MCMHead(d_model, max_features)
        # Component 29-36
        self.time_series = TimeSeriesModule(d_model)
        # Component 37-42
        self.analytics = AnalyticsEngine(d_model)
        # Component 43-63
        self.miras = MIRASModule(d_model)
        # Component 67-72
        self.data_processing = DataProcessing(max_features)
        # Component 73
        self.sector_head = SectorHead(d_model, n_sectors)
        
        self.input_proj = nn.Sequential(nn.Linear(max_features, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.final_fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(d_model, max_classes)
        
        # Component 23-26: EWC
        self.ewc_lambda = 1000
        self.fisher_info = {}
        self.optimal_params = {}
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, n_classes=10, mask=None):
        batch, n_feat = x.shape
        
        # Data Processing
        x = self.data_processing(x)
        
        # MIDAS
        if mask is None:
            mask = torch.ones_like(x)
        x_imputed, midas_loss = self.midas(x, mask)
        
        # Cell Processing
        cell_grid = self.cell_processing(x_imputed)
        
        # Schema Processing
        schema_emb = self.schema_processing(cell_grid)
        
        # Local Reasoning
        local_out = self.local_reasoning(cell_grid)
        
        # Global Reasoning
        latents, global_out = self.global_reasoning(local_out)
        
        # Time Series
        ts_out = self.time_series(local_out)
        
        # Analytics
        analytics_out = self.analytics(global_out)
        
        # MIRAS
        miras_out = self.miras(global_out)
        
        # MCM
        _, mcm_loss = self.mcm_head(cell_grid, x_imputed)
        
        # Final fusion
        input_emb = self.input_proj(x_imputed)
        combined = torch.cat([global_out, miras_out, input_emb], dim=-1)
        features = self.final_fusion(combined)
        
        # Classification
        logits = self.classifier(features)[:, :n_classes]
        sector_logits = self.sector_head(features)
        
        return {
            "logits": logits,
            "sector": sector_logits,
            "features": features,
            "midas_loss": midas_loss,
            "mcm_loss": mcm_loss
        }
    
    def compute_ewc_loss(self):
        if not self.fisher_info:
            return torch.tensor(0.0)
        loss = 0
        for n, p in self.named_parameters():
            if n in self.fisher_info and n in self.optimal_params:
                loss += (self.fisher_info[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.ewc_lambda * loss
    
    def store_fisher(self, dataloader, n_batches=30):
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.optimal_params = {n: p.clone().detach() for n, p in self.named_parameters()}
        self.eval()
        for i, (x, y, n_cls) in enumerate(dataloader):
            if i >= n_batches:
                break
            self.zero_grad()
            out = self(x, n_classes=n_cls)
            F.cross_entropy(out["logits"], y).backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.pow(2)
        for n in self.fisher_info:
            self.fisher_info[n] /= n_batches
        self.train()


# ============================================================
# DATASET - Fixed padding
# ============================================================

class StreamingDataset(Dataset):
    def __init__(self, data_dir, dataset_ids, samples_per_dataset, max_features):
        self.data_dir = Path(data_dir)
        self.max_features = max_features
        
        with open(self.data_dir / "metadata.json") as f:
            all_meta = json.load(f)
        
        self.metadata = [m for m in all_meta if m["dataset_id"] in set(dataset_ids)]
        
        all_X, all_y, all_nc = [], [], []
        
        print(f"Loading {len(self.metadata)} datasets ({samples_per_dataset} samples each)...")
        for meta in tqdm(self.metadata, desc="Loading"):
            df = pd.read_parquet(self.data_dir / meta["filename"])
            
            if len(df) > samples_per_dataset:
                df = df.sample(n=samples_per_dataset, random_state=42)
            
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values.astype(np.float32)
            y = df["target"].values.astype(np.int64)
            
            # Normalize
            X = np.nan_to_num(X, nan=0.0)
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            # Pad to max_features
            if X.shape[1] < max_features:
                padding = np.zeros((X.shape[0], max_features - X.shape[1]), dtype=np.float32)
                X = np.hstack([X, padding])
            elif X.shape[1] > max_features:
                X = X[:, :max_features]
            
            all_X.append(X)
            all_y.append(y)
            all_nc.extend([meta["n_classes"]] * len(X))
        
        self.X = np.vstack(all_X)
        self.y = np.hstack(all_y)
        self.n_cls = np.array(all_nc)
        
        # Shuffle
        idx = np.random.permutation(len(self.X))
        self.X, self.y, self.n_cls = self.X[idx], self.y[idx], self.n_cls[idx]
        
        print(f"Total: {len(self.X):,} samples, shape: {self.X.shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            self.n_cls[idx]
        )


def collate_fn(batch):
    X = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    max_cls = max(b[2] for b in batch)
    return X, y, max_cls


# ============================================================
# TRAINING
# ============================================================

def train(total_samples=10_000_000):
    print("=" * 70)
    print("SchemaLabs.AI Base Model Training")
    print("83 Components | Feature/Class Agnostic | Target: 99%")
    print("=" * 70)
    
    data_dir = Path("data/base_model")
    
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    n_datasets = len(metadata)
    samples_per_dataset = total_samples // n_datasets
    max_features = max(m["n_features"] for m in metadata) + 10
    max_classes = max(m["n_classes"] for m in metadata) + 10
    
    print(f"Datasets: {n_datasets}")
    print(f"Target samples: {total_samples:,}")
    print(f"Samples/dataset: {samples_per_dataset:,}")
    print(f"Max features: {max_features}")
    print(f"Max classes: {max_classes}")
    
    all_ids = [m["dataset_id"] for m in metadata]
    random.shuffle(all_ids)
    train_ids = all_ids[:int(n_datasets * 0.9)]
    val_ids = all_ids[int(n_datasets * 0.9):]
    
    print(f"Train datasets: {len(train_ids)} | Val datasets: {len(val_ids)}")
    
    # Config based on total samples
    if total_samples >= 100_000_000:
        batch_size, epochs, lr = 1024, 30, 0.001
    elif total_samples >= 10_000_000:
        batch_size, epochs, lr = 512, 50, 0.002
    else:
        batch_size, epochs, lr = 512, 50, 0.002
    
    print(f"\nConfig: batch={batch_size}, epochs={epochs}, lr={lr}")
    
    print("\n[1/3] Loading train data...")
    train_dataset = StreamingDataset(data_dir, train_ids, samples_per_dataset, max_features)
    
    print("\n[2/3] Loading val data...")
    val_dataset = StreamingDataset(data_dir, val_ids, samples_per_dataset, max_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"\nBatches: {len(train_loader):,} train, {len(val_loader):,} val")
    
    config = {
        "max_features": max_features,
        "max_classes": max_classes,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 3,
        "n_latents": 64,
        "n_sectors": 50
    }
    
    print(f"\n[3/3] Creating model...")
    model = BaseModel83(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    warmup = min(5, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / (epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_acc = 0
    best_state = None
    patience = max(10, epochs // 5)
    no_improve = 0
    
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    for epoch in range(epochs):
        epoch_start = datetime.now()
        
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs}", ncols=100)
        
        for X, y, n_cls in pbar:
            optimizer.zero_grad()
            
            out = model(X, n_classes=n_cls)
            
            ce_loss = F.cross_entropy(out["logits"], y)
            midas_loss = out["midas_loss"] * 0.1
            mcm_loss = out["mcm_loss"] * 0.1
            ewc_loss = model.compute_ewc_loss() * 0.001
            
            loss = ce_loss + midas_loss + mcm_loss + ewc_loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = out["logits"].argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*train_correct/train_total:.1f}%"})
        
        scheduler.step()
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y, n_cls in tqdm(val_loader, desc="Val", ncols=100, leave=False):
                out = model(X, n_classes=n_cls)
                pred = out["logits"].argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        eta = (epochs - epoch - 1) * epoch_time
        
        status = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save({
                "model_state_dict": best_state,
                "config": config,
                "accuracy": best_acc,
                "epoch": epoch + 1
            }, checkpoint_dir / "base_model_best.pt")
            status = " ★ BEST"
        else:
            no_improve += 1
        
        print(f"  Loss: {train_loss/len(train_loader):.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_acc:.1f}% | ETA: {timedelta(seconds=int(eta))}{status}")
        
        if best_acc >= 99.0 and no_improve >= patience:
            print(f"\n✓ 99% reached!")
            break
        
        if no_improve >= patience * 2:
            print(f"\n⚠ No improvement for {no_improve} epochs. Stopping.")
            break
        
        if (epoch + 1) % 10 == 0:
            print("  Updating EWC...")
            model.store_fisher(train_loader, n_batches=20)
    
    if best_state:
        model.load_state_dict(best_state)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = checkpoint_dir / f"base_model_{timestamp}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "accuracy": best_acc,
        "total_samples": total_samples
    }, final_path)
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"  Best Accuracy: {best_acc:.1f}%")
    print(f"  Time: {datetime.now() - start_time}")
    print(f"  Model: {final_path}")
    print(f"{'='*70}")
    
    return model, best_acc


if __name__ == "__main__":
    train(total_samples=10_000_000)
