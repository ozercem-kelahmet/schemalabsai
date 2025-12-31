import torch
import warnings; warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")
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
LOG_FILE = CHECKPOINT_DIR / 'schemalabsai_v1_log.txt'

feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                'recovery_index', 'response_rate']

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {msg}\n")

class MIDAS(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        return self.decoder(self.encoder(x))
    
    def impute(self, x, mask):
        recon = self.forward(x, mask)
        return x * mask + recon * (1 - mask)

class SectorModel(nn.Module):
    def __init__(self, n_sectors=50):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, n_sectors))
    
    def forward(self, x):
        return self.net(x)

class SubsectorModel(nn.Module):
    def __init__(self, n_sectors=50, n_subsectors=2500):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, 64)
        self.net = nn.Sequential(nn.Linear(74, 2048), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2048, n_subsectors))
    
    def forward(self, x, s):
        return self.net(torch.cat([x, self.emb(s)], 1))

def self_learning_batch(model_s, model_sub, X_unlab, threshold=0.9, batch_size=50000):
    model_s.eval()
    model_sub.eval()
    all_X, all_sec, all_sub = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(X_unlab), batch_size):
            batch = X_unlab[i:i+batch_size]
            sec_probs = torch.softmax(model_s(batch), dim=1)
            sec_conf, sec_pred = sec_probs.max(dim=1)
            
            sub_probs = torch.softmax(model_sub(batch, sec_pred), dim=1)
            sub_conf, sub_pred = sub_probs.max(dim=1)
            
            mask = (sec_conf * sub_conf) > threshold
            all_X.append(batch[mask])
            all_sec.append(sec_pred[mask])
            all_sub.append(sub_pred[mask])
    
    return torch.cat(all_X), torch.cat(all_sec), torch.cat(all_sub)

if __name__ == "__main__":
    log("=" * 60)
    log("SCHEMALABSAI V1 - BASE MODEL TRAINING")
    log("50 Sectors x 50 Subsectors = 2500 Classes")
    log("10M rows (200K per sector)")
    log("=" * 60)
    
    start_time = time.time()
    
    log("\n[1/6] Loading data...")
    all_X, all_sec, all_sub = [], [], []
    sector_to_id, subsector_to_id = {}, {}
    id_to_sector, id_to_subsector = {}, {}
    
    files = sorted(DATA_DIR.glob('*.parquet'))
    for f in tqdm(files, desc="Loading"):
        df = pd.read_parquet(f).sample(n=200000, random_state=42)
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
    
    log("\n[2/6] Initializing models...")
    midas = MIDAS(10, 256)
    model_s = SectorModel(n_sectors)
    model_sub = SubsectorModel(n_sectors, n_subsectors)
    
    total_params = sum(p.numel() for p in midas.parameters()) + sum(p.numel() for p in model_s.parameters()) + sum(p.numel() for p in model_sub.parameters())
    log(f"Total params: {total_params:,}")
    
    log("\n[3/6] MIDAS Pretraining (50 epochs)...")
    X_t = torch.FloatTensor(X_lab)
    opt_midas = AdamW(midas.parameters(), lr=0.001)
    
    for ep in tqdm(range(50), desc="MIDAS"):
        midas.train()
        indices = np.random.permutation(len(X_t))[:500000]
        batch = X_t[indices]
        mask = (torch.rand_like(batch) > 0.3).float()
        recon = midas(batch, mask)
        loss = ((recon - batch)**2 * (1-mask)).sum() / (1-mask).sum()
        opt_midas.zero_grad()
        loss.backward()
        opt_midas.step()
        
        if ep % 10 == 9:
            log(f"  MIDAS Ep {ep+1}: Loss={loss.item():.4f}")
    
    log("\n[4/6] Hierarchical Training (40 epochs)...")
    Xt = torch.FloatTensor(X_lab)
    yst = torch.LongTensor(y_sec_lab)
    ysub = torch.LongTensor(y_sub_lab)
    
    loader = DataLoader(TensorDataset(Xt, yst, ysub), batch_size=8192, shuffle=True)
    opt_s = AdamW(model_s.parameters(), lr=0.002)
    opt_sub = AdamW(model_sub.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for ep in tqdm(range(40), desc="Hierarchical"):
        model_s.train()
        model_sub.train()
        for bx, bs, bsub in loader:
            opt_s.zero_grad()
            loss_fn(model_s(bx), bs).backward()
            opt_s.step()
            opt_sub.zero_grad()
            loss_fn(model_sub(bx, bs), bsub).backward()
            opt_sub.step()
        
        if ep % 10 == 9:
            model_s.eval()
            model_sub.eval()
            with torch.no_grad():
                xt = Xt[:10000]
                sec_p = model_s(xt).argmax(-1)
                sec_acc = 100*(sec_p == yst[:10000]).float().mean()
                comb = 100*(model_sub(xt, sec_p).argmax(-1) == ysub[:10000]).float().mean()
            log(f"  Hier Ep {ep+1}: Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    log("\n[5/6] Self-Learning (5 rounds)...")
    X_unlab_t = torch.FloatTensor(X_unlab)
    
    for sl_round in tqdm(range(5), desc="Self-Learning"):
        X_pseudo, sec_pseudo, sub_pseudo = self_learning_batch(model_s, model_sub, X_unlab_t, threshold=0.85)
        
        if len(X_pseudo) < 1000:
            log(f"  SL Round {sl_round+1}: Only {len(X_pseudo)} samples, stopping")
            break
        
        X_comb = torch.cat([Xt, X_pseudo])
        y_sec_comb = torch.cat([yst, sec_pseudo])
        y_sub_comb = torch.cat([ysub, sub_pseudo])
        
        loader_sl = DataLoader(TensorDataset(X_comb, y_sec_comb, y_sub_comb), batch_size=8192, shuffle=True)
        
        model_s.train()
        model_sub.train()
        for _ in range(3):
            for bx, bs, bsub in loader_sl:
                opt_s.zero_grad()
                loss_fn(model_s(bx), bs).backward()
                opt_s.step()
                opt_sub.zero_grad()
                loss_fn(model_sub(bx, bs), bsub).backward()
                opt_sub.step()
        
        model_s.eval()
        model_sub.eval()
        with torch.no_grad():
            sec_p = model_s(X_unlab_t[:10000]).argmax(-1)
            sec_acc = 100*(sec_p == torch.LongTensor(y_sec_unlab[:10000])).float().mean()
            comb = 100*(model_sub(X_unlab_t[:10000], sec_p).argmax(-1) == torch.LongTensor(y_sub_unlab[:10000])).float().mean()
        
        log(f"  SL Round {sl_round+1}: +{len(X_pseudo):,} pseudo | Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    log("\n[6/6] Saving model...")
    checkpoint = {
        'midas': midas.state_dict(),
        'sector_model': model_s.state_dict(),
        'subsector_model': model_sub.state_dict(),
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
    
    save_path = CHECKPOINT_DIR / 'schemalabsai_v1.pt'
    torch.save(checkpoint, save_path)
    log(f"Saved: {save_path}")
    
    log("\n=== FINAL TEST ===")
    midas.eval()
    model_s.eval()
    model_sub.eval()
    
    X_test = torch.FloatTensor(X_unlab[:5000])
    y_test_sec = torch.LongTensor(y_sec_unlab[:5000])
    y_test_sub = torch.LongTensor(y_sub_unlab[:5000])
    
    for miss_rate in [0.0, 0.2, 0.4]:
        if miss_rate > 0:
            mask = (torch.rand_like(X_test) > miss_rate).float()
            X_imp = midas.impute(X_test * mask, mask)
        else:
            X_imp = X_test
        
        with torch.no_grad():
            sec_p = model_s(X_imp).argmax(-1)
            sec_acc = 100*(sec_p == y_test_sec).float().mean()
            comb = 100*(model_sub(X_imp, sec_p).argmax(-1) == y_test_sub).float().mean()
        
        log(f"  {int(miss_rate*100)}% missing: Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"SCHEMALABSAI V1 TAMAMLANDI!")
    log(f"Sure: {elapsed/60:.1f} dakika ({elapsed/3600:.2f} saat)")
    log(f"Model: {save_path}")
    log(f"{'='*60}")
