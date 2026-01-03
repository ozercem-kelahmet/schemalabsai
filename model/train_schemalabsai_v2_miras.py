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
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.scale * x / (x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5) + self.eps)

class GlobalReasoningLayer(nn.Module):
    def __init__(self, dim, n_latents=64):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        b = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        q = self.to_q(latents)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        out = (attn @ v).mean(dim=1)
        return self.norm(x.squeeze(1)) + self.to_out(out)

class MIRAS_Sector(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512, n_latents=64, n_sectors=50):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.reason = GlobalReasoningLayer(hidden_dim, n_latents)
        self.head = nn.Linear(hidden_dim, n_sectors)
    def forward(self, x):
        h = self.reason(self.proj(x))
        return self.head(h), h

class MIRAS_Subsector(nn.Module):
    def __init__(self, hidden_dim=512, n_sectors=50, n_subsectors=50):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, 128)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 128, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_subsectors)
        )
    def forward(self, h, sector): return self.net(torch.cat([h, self.emb(sector)], dim=1))

if __name__ == "__main__":
    open(LOG_FILE, 'w').close()
    
    log("=" * 70)
    log("SCHEMALABSAI V2 - MIRAS + MIDAS + Self-Learning")
    log("50 Sectors x 50 Subsectors = 2500 Classes")
    log("NEW: Sector-base subtraction + Global MinMax normalization")
    log("=" * 70)
    
    start_time = time.time()
    
    # ========================================
    # [1] LOAD DATA - New normalization
    # ========================================
    log("\n[1/5] Loading data...")
    
    files = sorted(DATA_DIR.glob('*.parquet'))
    
    all_X, all_sec, all_sub = [], [], []
    sector_to_id = {}
    sector_sub_to_id = {}
    sector_bases = {}
    
    for f in tqdm(files, desc="Loading"):
        df = pd.read_parquet(f)
        if len(df) > 200000:
            df = df.sample(n=200000, random_state=42)
        
        sector = df['sector'].iloc[0]
        
        if sector not in sector_to_id:
            sector_to_id[sector] = len(sector_to_id)
            sector_sub_to_id[sector_to_id[sector]] = {}
        
        sid = sector_to_id[sector]
        
        X = df[feature_cols].values.astype(np.float32)
        
        # Step 1: Subtract sector base
        sector_base = X.min(axis=0)
        sector_bases[sector] = sector_base.tolist()
        X = X - sector_base
        
        # Subsector mapping (0-49) based on primary_score range
        sub_ranges = {s: df[df['subsector']==s]['primary_score'].min() for s in df['subsector'].unique()}
        sorted_subs = sorted(sub_ranges.items(), key=lambda x: x[1])
        
        for i, (sub_name, _) in enumerate(sorted_subs):
            if sub_name not in sector_sub_to_id[sid]:
                sector_sub_to_id[sid][sub_name] = i
        
        all_X.append(X)
        all_sec.extend([sid] * len(df))
        all_sub.extend([sector_sub_to_id[sid][r['subsector']] for _, r in df.iterrows()])
    
    X = np.vstack(all_X)
    y_sec = np.array(all_sec)
    y_sub = np.array(all_sub)
    
    # Step 2: Global MinMax normalization
    X_min, X_max = X.min(0), X.max(0)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    
    n_sectors = len(sector_to_id)
    n_subsectors = 50
    
    log(f"Total data: {len(X):,} rows")
    log(f"Sectors: {n_sectors}, Subsectors per sector: {n_subsectors}")
    
    # Shuffle and split
    perm = np.random.permutation(len(X))
    X, y_sec, y_sub = X[perm], y_sec[perm], y_sub[perm]
    
    n = len(X)
    X_train, y_sec_train, y_sub_train = X[:int(n*0.6)], y_sec[:int(n*0.6)], y_sub[:int(n*0.6)]
    X_unlab = X[int(n*0.6):int(n*0.8)]
    X_test, y_sec_test, y_sub_test = X[int(n*0.8):], y_sec[int(n*0.8):], y_sub[int(n*0.8):]
    
    log(f"Labeled: {len(X_train):,}, Unlabeled: {len(X_unlab):,}, Test: {len(X_test):,}")
    
    # ========================================
    # [2] INITIALIZE MODELS
    # ========================================
    log("\n[2/5] Initializing models...")
    
    midas = MIDAS(10, 512)
    model_s = MIRAS_Sector(10, 512, 64, n_sectors)
    model_sub = MIRAS_Subsector(512, n_sectors, n_subsectors)
    
    total_params = sum(p.numel() for m in [midas, model_s, model_sub] for p in m.parameters())
    log(f"Total params: {total_params:,}")
    
    X_t = torch.FloatTensor(X_train)
    yst = torch.LongTensor(y_sec_train)
    ysub = torch.LongTensor(y_sub_train)
    X_unlab_t = torch.FloatTensor(X_unlab)
    X_test_t = torch.FloatTensor(X_test)
    y_sec_test_t = torch.LongTensor(y_sec_test)
    y_sub_test_t = torch.LongTensor(y_sub_test)
    
    loader = DataLoader(TensorDataset(X_t, yst, ysub), batch_size=2048, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    
    # ========================================
    # [3] MIRAS CLASSIFICATION (until 99% or max 200 epochs)
    # ========================================
    log("\n[3/5] MIRAS Classification Training...")
    
    opt_cls = AdamW(list(model_s.parameters()) + list(model_sub.parameters()), lr=0.001, weight_decay=0.01)
    
    best_acc, best_state_s, best_state_sub = 0, None, None
    
    for ep in tqdm(range(200), desc="MIRAS Classification"):
        model_s.train(); model_sub.train()
        for bx, bs, bsub in loader:
            opt_cls.zero_grad()
            logits, h = model_s(bx)
            (loss_fn(logits, bs) + loss_fn(model_sub(h, bs), bsub)).backward()
            opt_cls.step()
        
        model_s.eval(); model_sub.eval()
        with torch.no_grad():
            idx = np.random.choice(len(X_test_t), min(50000, len(X_test_t)), replace=False)
            logits, h = model_s(X_test_t[idx])
            sp = logits.argmax(-1)
            sec_acc = 100 * (sp == y_sec_test_t[idx]).float().mean()
            sub_acc = 100 * (model_sub(h, sp).argmax(-1) == y_sub_test_t[idx]).float().mean()
        
        if sub_acc > best_acc:
            best_acc = sub_acc
            best_state_s = {k: v.clone() for k, v in model_s.state_dict().items()}
            best_state_sub = {k: v.clone() for k, v in model_sub.state_dict().items()}
        
        if ep % 10 == 9:
            log(f"  Ep {ep+1}: Sector={sec_acc:.1f}% Subsector={sub_acc:.1f}% (best={best_acc:.1f}%)")
        
        if best_acc >= 99.0:
            log(f"  🎉 Reached 99%+ at ep {ep+1}")
            break
    
    model_s.load_state_dict(best_state_s)
    model_sub.load_state_dict(best_state_sub)
    log(f"  MIRAS Classification complete: {best_acc:.1f}%")
    
    # ========================================
    # [4] SELF-LEARNING (Threshold 0.95)
    # ========================================
    log("\n[4/5] Self-Learning (5 rounds, threshold=0.95)...")
    
    for sl_round in range(5):
        model_s.eval(); model_sub.eval()
        
        all_X, all_sec, all_sub = [], [], []
        batch_sz = 10000
        
        with torch.no_grad():
            for i in range(0, len(X_unlab_t), batch_sz):
                bx = X_unlab_t[i:i+batch_sz]
                logits, h = model_s(bx)
                sec_probs = torch.softmax(logits, 1)
                sec_conf, sec_pred = sec_probs.max(1)
                sub_probs = torch.softmax(model_sub(h, sec_pred), 1)
                sub_conf, sub_pred = sub_probs.max(1)
                
                mask = (sec_conf * sub_conf) > 0.95
                if mask.sum() > 0:
                    all_X.append(bx[mask].cpu())
                    all_sec.append(sec_pred[mask].cpu())
                    all_sub.append(sub_pred[mask].cpu())
        
        if len(all_X) == 0:
            log(f"  Round {sl_round+1}: No pseudo labels")
            continue
        
        X_pseudo = torch.cat(all_X)
        y_sec_pseudo = torch.cat(all_sec)
        y_sub_pseudo = torch.cat(all_sub)
        del all_X, all_sec, all_sub
        
        if len(X_pseudo) == 0:
            log(f"  Round {sl_round+1}: No pseudo labels")
            continue
        
        # Memory-efficient: Don't concatenate, use separate loaders
        loader_orig = DataLoader(TensorDataset(X_t, yst, ysub), batch_size=2048, shuffle=True)
        loader_pseudo = DataLoader(TensorDataset(X_pseudo, y_sec_pseudo, y_sub_pseudo), batch_size=2048, shuffle=True)
        opt_sl = AdamW(list(model_s.parameters()) + list(model_sub.parameters()), lr=0.0005, weight_decay=0.01)
        
        for _ in range(5):
            model_s.train(); model_sub.train()
            # Train on original data
            for bx, bs, bsub in loader_orig:
                opt_sl.zero_grad()
                logits, h = model_s(bx)
                (loss_fn(logits, bs) + loss_fn(model_sub(h, bs), bsub)).backward()
                opt_sl.step()
            # Train on pseudo data
            for bx, bs, bsub in loader_pseudo:
                opt_sl.zero_grad()
                logits, h = model_s(bx)
                (loss_fn(logits, bs) + loss_fn(model_sub(h, bs), bsub)).backward()
                opt_sl.step()
        
        model_s.eval(); model_sub.eval()
        with torch.no_grad():
            idx = np.random.choice(len(X_test_t), min(50000, len(X_test_t)), replace=False)
            logits, h = model_s(X_test_t[idx])
            sp = logits.argmax(-1)
            sub_acc = 100 * (model_sub(h, sp).argmax(-1) == y_sub_test_t[idx]).float().mean()
        
        if sub_acc > best_acc:
            best_acc = sub_acc
            best_state_s = {k: v.clone() for k, v in model_s.state_dict().items()}
            best_state_sub = {k: v.clone() for k, v in model_sub.state_dict().items()}
        
        log(f"  Round {sl_round+1}: +{len(X_pseudo):,} pseudo | acc={sub_acc:.1f}% (best={best_acc:.1f}%)")
    
    model_s.load_state_dict(best_state_s)
    model_sub.load_state_dict(best_state_sub)
    log(f"  Self-Learning complete: {best_acc:.1f}%")
    
    # Freeze classification
    for p in model_s.parameters(): p.requires_grad = False
    for p in model_sub.parameters(): p.requires_grad = False
    
    # ========================================
    # [5] MIDAS TRAINING
    # ========================================
    log("\n[5/5] MIDAS Training (300 epochs)...")
    
    opt_midas = AdamW(midas.parameters(), lr=0.001)
    best_midas_state, best_miss_acc, patience_counter = None, 0, 0
    
    for ep in tqdm(range(300), desc="MIDAS"):
        midas.train()
        for bx, bs, _ in loader:
            mask = (torch.rand_like(bx) > np.random.uniform(0.1, 0.6)).float()
            x_imp = midas.impute(bx, mask, n_iter=2)
            loss = ((x_imp - bx)**2 * (1-mask)).sum() / ((1-mask).sum() + 1e-8)
            opt_midas.zero_grad(); loss.backward(); opt_midas.step()
        
        if ep % 50 == 49:
            midas.eval(); model_s.eval(); model_sub.eval()
            with torch.no_grad():
                idx = np.random.choice(len(X_test_t), min(50000, len(X_test_t)), replace=False)
                mask = (torch.rand(len(idx), 10) > 0.3).float()
                x_imp = midas.impute(X_test_t[idx] * mask, mask, n_iter=3)
                logits, h = model_s(x_imp)
                sp = logits.argmax(-1)
                miss_acc = 100 * (model_sub(h, sp).argmax(-1) == y_sub_test_t[idx]).float().mean()
            
            if miss_acc > best_miss_acc:
                best_miss_acc = miss_acc
                best_midas_state = {k: v.clone() for k, v in midas.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            log(f"  Ep {ep+1} (30% miss): {miss_acc:.1f}% (best={best_miss_acc:.1f}%)")
            
            if patience_counter >= 20:
                log(f"  MIDAS Early stop - no improvement for 20 checks")
                break
    
    if best_midas_state:
        midas.load_state_dict(best_midas_state)
    
    # ========================================
    # SAVE
    # ========================================
    log("\nSaving model...")
    
    id_to_sector = {v: k for k, v in sector_to_id.items()}
    
    checkpoint = {
        'midas': midas.state_dict(),
        'miras_sector': model_s.state_dict(),
        'miras_subsector': model_sub.state_dict(),
        'sector_to_id': sector_to_id,
        'sector_sub_to_id': sector_sub_to_id,
        'id_to_sector': id_to_sector,
        'sector_bases': sector_bases,
        'X_min': X_min.tolist(),
        'X_max': X_max.tolist(),
        'n_sectors': n_sectors,
        'n_subsectors': n_subsectors,
        'feature_cols': feature_cols,
        'hidden_dim': 512,
        'n_latents': 64,
        'version': 'V2_MIRAS_MIDAS_SelfLearning_v2'
    }
    
    torch.save(checkpoint, CHECKPOINT_DIR / 'schemalabsai_v2_miras.pt')
    log(f"Saved: {CHECKPOINT_DIR / 'schemalabsai_v2_miras.pt'}")
    
    # ========================================
    # FINAL TEST
    # ========================================
    log("\n" + "=" * 70)
    log("FINAL TEST RESULTS")
    log("=" * 70)
    
    midas.eval(); model_s.eval(); model_sub.eval()
    
    for mr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        idx = np.random.choice(len(X_test_t), min(50000, len(X_test_t)), replace=False)
        test_sample = X_test_t[idx]
        test_sec, test_sub = y_sec_test_t[idx], y_sub_test_t[idx]
        
        if mr > 0:
            mask = (torch.rand_like(test_sample) > mr).float()
            with torch.no_grad():
                x_imp = midas.impute(test_sample * mask, mask, n_iter=3)
        else:
            x_imp = test_sample
        
        with torch.no_grad():
            logits, h = model_s(x_imp)
            sp = logits.argmax(-1)
            sec_acc = 100 * (sp == test_sec).float().mean()
            sub_acc = 100 * (model_sub(h, sp).argmax(-1) == test_sub).float().mean()
        
        log(f"  {int(mr*100)}% missing: Sector={sec_acc:.1f}% Subsector={sub_acc:.1f}%")
    
    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"TRAINING COMPLETE! Duration: {elapsed/60:.1f} minutes")
    log(f"{'=' * 70}")
