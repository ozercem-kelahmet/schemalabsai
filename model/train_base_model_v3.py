"""
SchemaLabs.AI Base Model v3 - Offset Labels
Her dataset'in label'ları unique - karışmaz
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
import sys
from datetime import datetime

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
        latents = self.norm1(latents + self.cross_attn(latents, x, x)[0])
        latents = self.norm2(latents + self.self_attn(latents, latents, latents)[0])
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
        loss = torch.tensor(0.0, device=x.device)
        if self.training and (1-mask).sum() > 0:
            loss = F.mse_loss(recon*(1-mask), x*(1-mask), reduction='sum') / ((1-mask).sum()+1e-8)
        return x*mask + recon*(1-mask), loss

class MCMHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, cell_grid, x_orig):
        batch, n_feat, d = cell_grid.shape
        mask = torch.rand(batch, n_feat, device=cell_grid.device) < 0.15
        pred = self.proj(cell_grid).squeeze(-1)
        loss = torch.tensor(0.0, device=cell_grid.device)
        if mask.any() and self.training:
            loss = F.mse_loss(pred[mask], x_orig[:,:n_feat][mask])
        return pred, loss

class TimeSeriesModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trend = nn.Linear(d_model, 1)
        self.seasonal = nn.Linear(d_model, 4)
        self.autocorr = nn.Linear(d_model, d_model)
    def forward(self, x):
        return {"trend": self.trend(x.mean(1)), "seasonal": self.seasonal(x.mean(1)), "autocorr": self.autocorr(x.mean(1))}

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
        total_classes = config["total_classes"]
        
        self.max_features = max_features
        self.d_model = d_model
        self.total_classes = total_classes
        self.n_sectors = n_sectors
        
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
        self.classifier = nn.Linear(d_model, total_classes)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x):
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
        sector_logits = self.sector_head(glob)
        inp = self.input_proj(x_imp)
        combined = torch.cat([glob, miras, inp], dim=-1)
        features = self.final_fusion(combined)
        logits = self.classifier(features)
        return {"logits": logits, "sector_logits": sector_logits, "features": features, "midas_loss": midas_loss, "mcm_loss": mcm_loss}


def load_dataset(filepath, max_features):
    df = pd.read_parquet(filepath)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    X = np.nan_to_num(X, nan=0.0)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    if X.shape[1] < max_features:
        X = np.hstack([X, np.zeros((X.shape[0], max_features - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > max_features:
        X = X[:, :max_features]
    return X, y


def train(data_dir="data/base_model", max_samples=1_000_000):
    print("=" * 60)
    print("SchemaLabs.AI Base Model v3 - Offset Labels")
    print("Her dataset'in label'ları unique")
    print("=" * 60)
    
    with open(Path(data_dir) / "metadata.json") as f:
        metadata = json.load(f)
    
    max_features = max(m["n_features"] for m in metadata) + 10
    
    # 1M sample için dataset seç
    total_rows = 0
    selected_meta = []
    for m in metadata:
        if total_rows >= max_samples:
            break
        selected_meta.append(m)
        total_rows += m["n_rows"]
    
    # Offset hesapla - her dataset için unique label aralığı
    offset = 0
    for m in selected_meta:
        m["label_offset"] = offset
        offset += m["n_classes"]
    
    total_classes = offset
    
    print(f"Datasets: {len(selected_meta)}, Rows: ~{total_rows:,}")
    print(f"Max features: {max_features}, Total classes: {total_classes}")
    
    # Train/val split
    random.shuffle(selected_meta)
    n_train = int(len(selected_meta) * 0.9)
    train_meta = selected_meta[:n_train]
    val_meta = selected_meta[n_train:]
    
    print(f"Train: {len(train_meta)}, Val: {len(val_meta)}")
    
    config = {
        "max_features": max_features, 
        "total_classes": total_classes, 
        "d_model": 256, 
        "n_heads": 8, 
        "n_layers": 3, 
        "n_latents": 64, 
        "n_sectors": 50
    }
    
    model = BaseModel83(config).to(device)
    model = torch.compile(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    lr = 0.001
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    epochs = 20
    warmup_epochs = min(3, epochs // 5)
    
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    gradient_accumulation_steps = 2
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    batch_size = 256
    best_val_acc = 0
    patience = 5
    no_improve = 0
    
    start_time = datetime.now()
    
    for epoch in range(epochs):
        model.train()
        epoch_correct, epoch_total = 0, 0
        epoch_loss = 0
        batch_idx = 0
        
        random.shuffle(train_meta)
        
        for i, meta in enumerate(train_meta):
            X, y = load_dataset(Path(data_dir) / meta["filename"], max_features)
            
            # Label offset uygula
            y = y + meta["label_offset"]
            
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
            
            for j in range(0, len(X), batch_size):
                X_batch = torch.FloatTensor(X[j:j+batch_size]).to(device)
                y_batch = torch.LongTensor(y[j:j+batch_size]).to(device)
                
                # Data augmentation
                if np.random.random() > 0.5:
                    noise = torch.randn_like(X_batch) * 0.01
                    X_batch = X_batch + noise
                
                with autocast('cuda'):
                    out = model(X_batch)
                    ce_loss = F.cross_entropy(out["logits"], y_batch)
                    sector_targets = y_batch % model.n_sectors
                    sector_loss = F.cross_entropy(out["sector_logits"], sector_targets)
                    loss = ce_loss + sector_loss + out["midas_loss"]*0.1 + out["mcm_loss"]*0.1
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                epoch_correct += (out["logits"].argmax(1) == y_batch).sum().item()
                epoch_total += len(y_batch)
                epoch_loss += loss.item() * gradient_accumulation_steps
                batch_idx += 1
            
            # Progress
            acc = 100*epoch_correct/epoch_total if epoch_total > 0 else 0
            avg_loss = epoch_loss / batch_idx if batch_idx > 0 else 0
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = epoch_total / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} | {i+1}/{len(train_meta)} | acc: {acc:.1f}% | loss: {avg_loss:.3f} | {speed:.0f} s/s")
            sys.stdout.flush()
        
        scheduler.step()
        train_acc = 100 * epoch_correct / epoch_total
        print()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for meta in val_meta:
                X, y = load_dataset(Path(data_dir) / meta["filename"], max_features)
                y = y + meta["label_offset"]  # Offset uygula
                
                X_t = torch.FloatTensor(X).to(device)
                y_t = torch.LongTensor(y).to(device)
                
                for j in range(0, len(X_t), batch_size*4):
                    X_b = X_t[j:j+batch_size*4]
                    y_b = y_t[j:j+batch_size*4]
                    out = model(X_b)
                    val_correct += (out["logits"].argmax(1) == y_b).sum().item()
                    val_total += len(y_b)
        
        val_acc = 100 * val_correct / val_total
        
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(), 
                "config": config, 
                "accuracy": best_val_acc,
                "offsets": {m["filename"]: m["label_offset"] for m in selected_meta}
            }, checkpoint_dir / "base_model_best.pt")
            status = " ★"
        else:
            no_improve += 1
        
        print(f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%{status}")
        
        if best_val_acc >= 99.0:
            print("🎉 99% reached!")
            break
        
        if no_improve >= patience:
            print(f"Early stop - no improve for {patience} epochs")
            break
    
    print(f"\nDone! Best: {best_val_acc:.1f}% | Time: {datetime.now()-start_time}")
    torch.save({
        "model_state_dict": model.state_dict(), 
        "config": config, 
        "accuracy": best_val_acc,
        "offsets": {m["filename"]: m["label_offset"] for m in selected_meta}
    }, checkpoint_dir / "base_model_final.pt")


if __name__ == "__main__":
    train(data_dir="data/base_model", max_samples=1_000_000)
