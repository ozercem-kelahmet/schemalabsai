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

class MLPMemory(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.norm = RMSNorm(dim)
    def forward(self, x):
        h = silu(self.w1(x)) * self.w3(x)
        return self.norm(x + self.w2(h))

class GlobalReasoningLayer(nn.Module):
    def __init__(self, dim, n_latents=64, n_heads=4):
        super().__init__()
        self.n_latents = n_latents
        self.dim = dim
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.cross_attn_q = nn.Linear(dim, dim)
        self.cross_attn_k = nn.Linear(dim, dim)
        self.cross_attn_v = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.gate = nn.Parameter(torch.zeros(1))
        self.mlp_memory = MLPMemory(dim, expansion=4)
        self.norm = RMSNorm(dim)
        self.memory_alpha = 0.7
        self.persistent_memory = None
    
    def forward(self, x):
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        if self.persistent_memory is not None and self.persistent_memory.shape[0] == B:
            latents = self.memory_alpha * latents + (1 - self.memory_alpha) * self.persistent_memory
        Q = self.cross_attn_q(latents).view(B, self.n_latents, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.cross_attn_k(x.unsqueeze(1)).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.cross_attn_v(x.unsqueeze(1)).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, self.n_latents, self.dim)
        gate = torch.sigmoid(self.gate)
        out = gate * out + (1 - gate) * latents
        out = self.mlp_memory(out)
        self.persistent_memory = out.detach()
        pooled = out.mean(dim=1)
        return self.norm(pooled)

class MIRASModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, n_sectors=50, n_subsectors=2500, n_latents=64):
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
        sector_logits = self.sector_head(h)
        return sector_logits, h
    
    def predict_subsector(self, h, sector_ids):
        emb = self.subsector_emb(sector_ids)
        return self.subsector_head(torch.cat([h, emb], dim=1))

def self_learning_batch(model, X_unlab, threshold=0.9, batch_size=50000):
    model.eval()
    all_X, all_sec, all_sub = [], [], []
    with torch.no_grad():
        for i in range(0, len(X_unlab), batch_size):
            batch = X_unlab[i:i+batch_size]
            sec_logits, h = model(batch)
            sec_probs = torch.softmax(sec_logits, dim=1)
            sec_conf, sec_pred = sec_probs.max(dim=1)
            sub_logits = model.predict_subsector(h, sec_pred)
            sub_probs = torch.softmax(sub_logits, dim=1)
            sub_conf, sub_pred = sub_probs.max(dim=1)
            mask = (sec_conf * sub_conf) > threshold
            all_X.append(batch[mask])
            all_sec.append(sec_pred[mask])
            all_sub.append(sub_pred[mask])
    return torch.cat(all_X), torch.cat(all_sec), torch.cat(all_sub)

if __name__ == "__main__":
    log("=" * 60)
    log("SCHEMALABSAI V2 - MIRAS + MIDAS + SL")
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
    model = MIRASModel(10, 256, n_sectors, n_subsectors, n_latents=64)
    midas_params = sum(p.numel() for p in midas.parameters())
    miras_params = sum(p.numel() for p in model.parameters())
    log(f"MIDAS: {midas_params:,}, MIRAS: {miras_params:,}, Total: {midas_params+miras_params:,}")
    
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
    
    log("\n[4/6] MIRAS Training (40 epochs)...")
    Xt = torch.FloatTensor(X_lab)
    yst = torch.LongTensor(y_sec_lab)
    ysub = torch.LongTensor(y_sub_lab)
    loader = DataLoader(TensorDataset(Xt, yst, ysub), batch_size=4096, shuffle=True)
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for ep in tqdm(range(20), desc="MIRAS"):
        model.train()
        for bx, bs, bsub in loader:
            opt.zero_grad()
            sec_logits, h = model(bx)
            loss = loss_fn(sec_logits, bs) + loss_fn(model.predict_subsector(h, bs), bsub)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 10 == 9:
            model.eval()
            with torch.no_grad():
                xt = Xt[:10000]
                sec_logits, h = model(xt)
                sec_acc = 100*(sec_logits.argmax(-1) == yst[:10000]).float().mean()
                comb = 100*(model.predict_subsector(h, sec_logits.argmax(-1)).argmax(-1) == ysub[:10000]).float().mean()
            log(f"  MIRAS Ep {ep+1}: Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    log("\n[5/6] Self-Learning (5 rounds)...")
    X_unlab_t = torch.FloatTensor(X_unlab)
    for sl_round in tqdm(range(5), desc="SL"):
        X_pseudo, sec_pseudo, sub_pseudo = self_learning_batch(model, X_unlab_t, threshold=0.85)
        if len(X_pseudo) < 1000:
            log(f"  SL {sl_round+1}: Only {len(X_pseudo)}, stopping")
            break
        X_comb = torch.cat([Xt, X_pseudo])
        y_sec_comb = torch.cat([yst, sec_pseudo])
        y_sub_comb = torch.cat([ysub, sub_pseudo])
        loader_sl = DataLoader(TensorDataset(X_comb, y_sec_comb, y_sub_comb), batch_size=4096, shuffle=True)
        model.train()
        for _ in range(3):
            for bx, bs, bsub in loader_sl:
                opt.zero_grad()
                sec_logits, h = model(bx)
                loss = loss_fn(sec_logits, bs) + loss_fn(model.predict_subsector(h, bs), bsub)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            sec_logits, h = model(X_unlab_t[:10000])
            comb = 100*(model.predict_subsector(h, sec_logits.argmax(-1)).argmax(-1) == torch.LongTensor(y_sec_unlab[:10000])).float().mean()
        log(f"  SL {sl_round+1}: +{len(X_pseudo):,} | Combined={comb:.1f}%")
    
    log("\n[6/6] Saving...")
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
    midas.eval()
    model.eval()
    X_test = torch.FloatTensor(X_unlab[:5000])
    y_test_sec = torch.LongTensor(y_sec_unlab[:5000])
    y_test_sub = torch.LongTensor(y_sub_unlab[:5000])
    
    for miss in [0.0, 0.2, 0.4]:
        if miss > 0:
            mask = (torch.rand_like(X_test) > miss).float()
            X_imp = midas.impute(X_test * mask, mask)
        else:
            X_imp = X_test
        with torch.no_grad():
            sec_logits, h = model(X_imp)
            sec_pred = sec_logits.argmax(-1)
            sec_acc = 100*(sec_pred == y_test_sec).float().mean()
            comb = 100*(model.predict_subsector(h, sec_pred).argmax(-1) == y_test_sub).float().mean()
        log(f"  {int(miss*100)}% missing: Sector={sec_acc:.1f}% Combined={comb:.1f}%")
    
    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"V2 MIRAS TAMAMLANDI! Sure: {elapsed/60:.1f} dk")
    log(f"{'='*60}")
