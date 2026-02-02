"""
SchemaLabs.AI Base Model v5 - Her dataset sıfırdan fine-tune
Fine-tune ile aynı: Her dataset için model sıfırdan
"""
import warnings
warnings.filterwarnings("ignore")

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
import sys
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def get_dynamic_config(n_samples, n_features, n_classes):
    complexity = (n_samples * n_classes) / 1000
    
    if n_samples < 100:
        batch_size = 4
    elif n_samples < 500:
        batch_size = 16
    elif n_samples < 2000:
        batch_size = 32
    else:
        batch_size = 64
    
    if n_samples < 100:
        epochs = 100
    elif n_samples < 500:
        epochs = 50
    elif n_samples < 2000:
        epochs = 30
    else:
        epochs = 20
    
    if n_classes > 20:
        epochs = min(epochs + 10, 100)
    
    patience = max(5, min(25, epochs // 4))
    
    if n_samples < 500:
        lr = 0.005
    elif n_samples < 2000:
        lr = 0.003
    else:
        lr = 0.001
    
    return {'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'patience': patience}


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
    def __init__(self, d_input, d_hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_input*2, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden//2))
        self.decoder = nn.Sequential(nn.Linear(d_hidden//2, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_input))
        self.d_input = d_input
    def forward(self, x, mask=None):
        if x.shape[1] < self.d_input:
            x = F.pad(x, (0, self.d_input - x.shape[1]))
        elif x.shape[1] > self.d_input:
            x = x[:, :self.d_input]
        if mask is None:
            mask = torch.ones_like(x)
        return x, torch.tensor(0.0, device=x.device)

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
            target = x_orig[:, :n_feat] if x_orig.shape[1] >= n_feat else F.pad(x_orig, (0, n_feat - x_orig.shape[1]))
            loss = F.mse_loss(pred[mask], target[mask])
        return pred, loss

class TimeSeriesModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trend = nn.Linear(d_model, 1)
    def forward(self, x):
        return {"trend": self.trend(x.mean(1))}

class AnalyticsEngine(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.stats = nn.Linear(d_model, 8)
    def forward(self, x):
        return {"stats": self.stats(x)}

class MIRASModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.retention_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.low_rank_down = nn.Linear(d_model, 32)
        self.low_rank_up = nn.Linear(32, d_model)
    def forward(self, x):
        x = x * self.retention_gate(x)
        x = x + self.low_rank_up(self.low_rank_down(x)) * 0.1
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
        n_classes = config.get("n_classes", 10)
        
        self.max_features = max_features
        self.d_model = d_model
        self.n_sectors = n_sectors
        self.n_classes = n_classes
        
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
        
        self.values_proj = nn.Sequential(
            nn.Linear(max_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.final_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.clamp(x, -5.0, 5.0)
        
        if x.shape[1] < self.max_features:
            x = F.pad(x, (0, self.max_features - x.shape[1]))
        elif x.shape[1] > self.max_features:
            x = x[:, :self.max_features]
        
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
        
        values_emb = self.values_proj(x_imp)
        combined = torch.cat([glob, values_emb], dim=-1)
        
        output = self.final_head(combined)
        
        return {
            "output": output, 
            "sector": sector_logits, 
            "midas_loss": midas_loss, 
            "mcm_loss": mcm_loss
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


def finetune_one_dataset(X, y, n_classes, n_features, max_features, device):
    """Her dataset için model SIFIRDAN - fine-tune gibi"""
    
    # Model sıfırdan oluştur - fine-tune gibi
    config = {
        "max_features": max_features, 
        "d_model": 256, 
        "n_heads": 8, 
        "n_layers": 3, 
        "n_latents": 64, 
        "n_sectors": 50,
        "n_classes": n_classes
    }
    model = BaseModel83(config).to(device)
    
    # Dynamic config
    cfg = get_dynamic_config(len(X), n_features, n_classes)
    batch_size = cfg['batch_size']
    max_epochs = cfg['epochs']
    lr = cfg['lr']
    patience = cfg['patience']
    
    # Train/Val split
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.9)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Optimizer + Scheduler (fine-tune gibi)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_epochs = min(3, max_epochs // 5)
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = GradScaler('cuda')
    gradient_accumulation_steps = 2
    
    best_val_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_correct, train_total = 0, 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            if random.random() > 0.5:
                batch_X = batch_X + torch.randn_like(batch_X) * 0.01
            
            with autocast('cuda'):
                out = model(batch_X)
                logits = out["output"]
                sector_targets = batch_y % model.n_sectors
                sector_loss = F.cross_entropy(out["sector"], sector_targets)
                mcm_loss = out.get("mcm_loss", torch.tensor(0.0, device=device))
                loss = F.cross_entropy(logits, batch_y) + sector_loss + 0.1 * mcm_loss
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)
        
        scheduler.step()
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        with torch.no_grad():
            out = model(X_val_t)
            val_correct = (out["output"].argmax(1) == y_val_t).sum().item()
            val_acc = 100 * val_correct / len(y_val_t)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            status = "★"
        else:
            no_improve += 1
            status = ""
        
        print(f"    Ep {epoch+1}/{max_epochs}: train {train_acc:.1f}% | val {val_acc:.1f}% | best {best_val_acc:.1f}%{status}")
        
        if best_val_acc >= 99.0:
            break
        
        if no_improve >= patience:
            break
    
    return best_val_acc, best_state, config


def train(data_dir="data/base_model", max_samples=1_000_000):
    print("=" * 60)
    print("SchemaLabs.AI Base Model v5 - Sıfırdan Fine-tune")
    print("Her dataset için model sıfırdan - fine-tune gibi")
    print("=" * 60)
    
    with open(Path(data_dir) / "metadata.json") as f:
        metadata = json.load(f)
    
    max_features = max(m["n_features"] for m in metadata) + 10
    
    total_rows = 0
    selected_meta = []
    for m in metadata:
        if total_rows >= max_samples:
            break
        selected_meta.append(m)
        total_rows += m["n_rows"]
    
    print(f"Datasets: {len(selected_meta)}, Rows: ~{total_rows:,}")
    print(f"Max features: {max_features}")
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_time = datetime.now()
    accuracies = []
    all_states = []
    
    for i, meta in enumerate(selected_meta):
        print(f"\n[{i+1}/{len(selected_meta)}] {meta['filename']} - {meta['n_classes']} classes, {meta['n_rows']} rows, {meta['n_features']} features")
        
        X, y = load_dataset(Path(data_dir) / meta["filename"])
        n_classes = meta["n_classes"]
        n_features = meta["n_features"]
        
        val_acc, best_state, config = finetune_one_dataset(X, y, n_classes, n_features, max_features, device)
        accuracies.append(val_acc)
        all_states.append(best_state)
        
        avg_acc = sum(accuracies) / len(accuracies)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"  => Val: {val_acc:.1f}% | Avg: {avg_acc:.1f}% | Time: {elapsed:.0f}s")
        
        if (i + 1) % 10 == 0:
            torch.save({
                "config": config,
                "avg_accuracy": avg_acc,
                "datasets_trained": i + 1,
                "accuracies": accuracies
            }, checkpoint_dir / "base_model_checkpoint.pt")
    
    final_avg = sum(accuracies) / len(accuracies)
    print(f"\n{'='*60}")
    print(f"Done! Avg Val Accuracy: {final_avg:.1f}%")
    print(f"Time: {datetime.now()-start_time}")
    print(f"{'='*60}")
    
    # En iyi accuracy'ye sahip model'in state'ini kaydet
    best_idx = accuracies.index(max(accuracies))
    torch.save({
        "model_state_dict": all_states[best_idx],
        "config": config,
        "avg_accuracy": final_avg,
        "best_accuracy": max(accuracies),
        "datasets_trained": len(selected_meta),
        "accuracies": accuracies
    }, checkpoint_dir / "base_model_final.pt")


if __name__ == "__main__":
    train(data_dir="data/base_model", max_samples=1_000_000)
