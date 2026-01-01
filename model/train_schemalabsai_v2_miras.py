import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

DATA_DIR = Path('../data/training_50x50')
CHECKPOINT_DIR = Path('../checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_FILE = CHECKPOINT_DIR / 'schemalabsai_v2_miras_log.txt'

feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                'recovery_index', 'response_rate']

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {msg}\n")

def silu(x):
    return x * torch.sigmoid(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)

class MIDAS(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x, mask=None):
        if mask is not None: x = x * mask
        return self.decoder(self.encoder(x))
    def impute(self, x, mask):
        return x * mask + self.forward(x, mask) * (1 - mask)

class MLPMemory(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class GlobalReasoningLayer(nn.Module):
    def __init__(self, dim, n_latents=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLPMemory(dim)
    
    def forward(self, x):
        b = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1: x = x.unsqueeze(0)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        q = self.to_q(latents)
        x_for_kv = x.unsqueeze(1) if x.dim() == 2 else x
        kv = self.to_kv(x_for_kv)
        k, v = kv.chunk(2, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = attn @ v
        out = out.mean(dim=1)
        x_out = x.squeeze(1) if x.dim() == 3 and x.shape[1] == 1 else x
        if x_out.dim() == 3: x_out = x_out.mean(dim=1)
        x_out = x_out + self.gate * self.to_out(out)
        x_out = self.norm1(x_out)
        x_out = x_out + self.mlp(x_out)
        x_out = self.norm2(x_out)
        return x_out

class MIRASModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, n_latents=64, n_sectors=50, n_subsectors=2500):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.reasoning = GlobalReasoningLayer(hidden_dim, n_latents=n_latents)
        self.sector_head = nn.Linear(hidden_dim, n_sectors)
        self.subsector_emb = nn.Embedding(n_sectors, 64)
        self.subsector_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, n_subsectors)
        )
    
    def forward(self, x):
        h = self.input_proj(x)
        h = self.reasoning(h)
        return self.sector_head(h), h
    
    def predict_subsector(self, h, sector_ids):
        emb = self.subsector_emb(sector_ids)
        return self.subsector_head(torch.cat([h, emb], dim=1))

def self_learning_round(midas, model, X_unlab, threshold=0.7, batch_size=50000):
    midas.eval(); model.eval()
    all_X, all_sec, all_sub = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(X_unlab), batch_size):
            batch = X_unlab[i:i+batch_size]
            mask = (torch.rand_like(batch) > 0.3).float()
            X_imp = midas.impute(batch * mask, mask)
            
            sec_logits, h = model(X_imp)
            sec_probs = torch.softmax(sec_logits, dim=1)
            sec_conf, sec_pred = sec_probs.max(dim=1)
            
            sub_logits = model.predict_subsector(h, sec_pred)
            sub_probs = torch.softmax(sub_logits, dim=1)
            sub_conf, sub_pred = sub_probs.max(dim=1)
            
            high_conf = (sec_conf * sub_conf) > threshold
            all_X.append(X_imp[high_conf])
            all_sec.append(sec_pred[high_conf])
            all_sub.append(sub_pred[high_conf])
    
    return torch.cat(all_X), torch.cat(all_sec), torch.cat(all_sub)

if __name__ == "__main__":
    log("=" * 60)
    log("SCHEMALABSAI V2 - MIRAS + MIDAS + SELF-LEARNING")
    log("50 Sectors x 50 Subsectors = 2500 Classes")
    log("=" * 60)
    
    start_time = time.time()
    
    log("\n[1/5] Loading data...")
    all_X, all_sec, all_sub = [], [], []
    sector_to_id, subsector_to_id = {}, {}
    id_to_sector, id_to_subsector = {}, {}
    
    files = sorted(DATA_DIR.glob('*.parquet'))
    for f in tqdm(files, desc="Loading"):
        df = pd.read_parquet(f).sample(n=min(200000, len(pd.read_parquet(f))), random_state=42)
        sector = df['sector'].iloc[0]
        if sector not in sector_to_id:
            sid = len(sector_to_id)
            sector_to_id[sector] = sid
            id_to_sector[sid] = sector
        for sub in df['subsector'].unique():
            key = f"{sector}_{sub}"
            if key not in subsector_to_id:
                subid = len(subsector_to_id)
                subsector_to_id[key] = subid
                id_to_subsector[subid] = key
        X = df[feature_cols].values.astype(np.float32)
        all_X.append(X)
        all_sec.extend([sector_to_id[sector]] * len(df))
        all_sub.extend([subsector_to_id[f"{sector}_{r['subsector']}"] for _, r in df.iterrows()])
    
    X = np.vstack(all_X)
    mean, std = X.mean(0), X.std(0) + 1e-8
    X_norm = (X - mean) / std
    y_sec, y_sub = np.array(all_sec), np.array(all_sub)
    
    perm = np.random.permutation(len(X))
    X_norm, y_sec, y_sub = X_norm[perm], y_sec[perm], y_sub[perm]
    
    n_sectors = len(sector_to_id)
    n_subsectors = len(subsector_to_id)
    log(f"Data: {len(X):,} rows, {n_sectors} sectors, {n_subsectors} subsectors")
    
    split = int(len(X) * 0.8)
    X_lab, y_sec_lab, y_sub_lab = X_norm[:split], y_sec[:split], y_sub[:split]
    X_unlab = X_norm[split:]
    y_sec_unlab, y_sub_unlab = y_sec[split:], y_sub[split:]
    log(f"Labeled: {len(X_lab):,}, Unlabeled: {len(X_unlab):,}")
    
    log("\n[2/5] Initializing models...")
    midas = MIDAS(10, 256)
    model = MIRASModel(10, 256, 64, n_sectors, n_subsectors)
    
    midas_params = sum(p.numel() for p in midas.parameters())
    miras_params = sum(p.numel() for p in model.parameters())
    log(f"MIDAS: {midas_params:,}, MIRAS: {miras_params:,}, Total: {midas_params + miras_params:,}")
    
    log("\n[3/5] Joint Training (40 epochs)...")
    X_t = torch.FloatTensor(X_lab)
    yst = torch.LongTensor(y_sec_lab)
    ysub = torch.LongTensor(y_sub_lab)
    
    loader = DataLoader(TensorDataset(X_t, yst, ysub), batch_size=8192, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = AdamW(list(midas.parameters()) + list(model.parameters()), lr=0.001)
    
    for ep in tqdm(range(40), desc="Joint Training"):
        midas.train(); model.train()
        for bx, bs, bsub in loader:
            if np.random.random() > 0.5:
                miss_rate = np.random.uniform(0.1, 0.5)
                mask = (torch.rand_like(bx) > miss_rate).float()
                recon = midas(bx, mask)
                bx_input = bx * mask + recon * (1 - mask)
                recon_loss = ((recon - bx)**2 * (1-mask)).sum() / ((1-mask).sum() + 1e-8)
            else:
                bx_input = bx
                recon_loss = torch.tensor(0.0)
            
            sec_logits, h = model(bx_input)
            sub_logits = model.predict_subsector(h, bs)
            cls_loss = loss_fn(sec_logits, bs) + loss_fn(sub_logits, bsub)
            
            opt.zero_grad()
            (recon_loss + cls_loss).backward()
            opt.step()
        
        if ep % 10 == 9:
            midas.eval(); model.eval()
            with torch.no_grad():
                test_X = torch.FloatTensor(X_unlab[:10000])
                sec_logits, h = model(test_X)
                sec_pred = sec_logits.argmax(-1)
                sec_acc = 100 * (sec_pred == torch.LongTensor(y_sec_unlab[:10000])).float().mean()
                comb = 100 * (model.predict_subsector(h, sec_pred).argmax(-1) == torch.LongTensor(y_sub_unlab[:10000])).float().mean()
            log(f"  Ep {ep+1}: Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    log("\n[4/5] Self-Learning (5 rounds)...")
    X_unlab_t = torch.FloatTensor(X_unlab)
    
    for sl_round in tqdm(range(5), desc="Self-Learning"):
        X_pseudo, y_sec_pseudo, y_sub_pseudo = self_learning_round(midas, model, X_unlab_t, threshold=0.7)
        
        if len(X_pseudo) == 0:
            log(f"  Round {sl_round+1}: No pseudo labels")
            continue
        
        X_comb = torch.cat([X_t, X_pseudo])
        y_sec_comb = torch.cat([yst, y_sec_pseudo])
        y_sub_comb = torch.cat([ysub, y_sub_pseudo])
        
        loader_sl = DataLoader(TensorDataset(X_comb, y_sec_comb, y_sub_comb), batch_size=8192, shuffle=True)
        
        for _ in range(5):
            midas.train(); model.train()
            for bx, bs, bsub in loader_sl:
                if np.random.random() > 0.5:
                    miss_rate = np.random.uniform(0.1, 0.5)
                    mask = (torch.rand_like(bx) > miss_rate).float()
                    recon = midas(bx, mask)
                    bx_input = bx * mask + recon * (1 - mask)
                    recon_loss = ((recon - bx)**2 * (1-mask)).sum() / ((1-mask).sum() + 1e-8)
                else:
                    bx_input = bx
                    recon_loss = torch.tensor(0.0)
                
                sec_logits, h = model(bx_input)
                cls_loss = loss_fn(sec_logits, bs) + loss_fn(model.predict_subsector(h, bs), bsub)
                opt.zero_grad()
                (recon_loss + cls_loss).backward()
                opt.step()
        
        midas.eval(); model.eval()
        with torch.no_grad():
            test_X = torch.FloatTensor(X_unlab[:10000])
            sec_logits, h = model(test_X)
            sec_pred = sec_logits.argmax(-1)
            comb = 100 * (model.predict_subsector(h, sec_pred).argmax(-1) == torch.LongTensor(y_sub_unlab[:10000])).float().mean()
        log(f"  Round {sl_round+1}: +{len(X_pseudo):,} pseudo | Combined={comb:.1f}%")
    
    log("\n[5/5] Saving model...")
    checkpoint = {
        'midas': midas.state_dict(),
        'miras_model': model.state_dict(),
        'sector_to_id': sector_to_id,
        'subsector_to_id': subsector_to_id,
        'id_to_sector': id_to_sector,
        'id_to_subsector': id_to_subsector,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_sectors': n_sectors,
        'n_subsectors': n_subsectors,
        'feature_cols': feature_cols
    }
    save_path = CHECKPOINT_DIR / 'schemalabsai_v2_miras.pt'
    torch.save(checkpoint, save_path)
    log(f"Saved: {save_path}")
    
    log("\n=== FINAL TEST ===")
    midas.eval(); model.eval()
    test_X = torch.FloatTensor(X_unlab[:5000])
    test_sec = torch.LongTensor(y_sec_unlab[:5000])
    test_sub = torch.LongTensor(y_sub_unlab[:5000])
    
    for miss_rate in [0.0, 0.2, 0.4, 0.6]:
        if miss_rate > 0:
            mask = (torch.rand_like(test_X) > miss_rate).float()
            with torch.no_grad():
                X_imp = midas.impute(test_X * mask, mask)
        else:
            X_imp = test_X
        
        with torch.no_grad():
            sec_logits, h = model(X_imp)
            sec_pred = sec_logits.argmax(-1)
            sec_acc = 100 * (sec_pred == test_sec).float().mean()
            comb = 100 * (model.predict_subsector(h, sec_pred).argmax(-1) == test_sub).float().mean()
        log(f"  {int(miss_rate*100)}% missing: Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"TAMAMLANDI! Sure: {elapsed/60:.1f} dk")
    log(f"{'='*60}")
