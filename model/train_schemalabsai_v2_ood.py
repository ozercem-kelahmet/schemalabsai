import torch
import torch.nn as nn
import torch.nn.functional as F
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
LOG_FILE = CHECKPOINT_DIR / 'schemalabsai_v2_ood_log.txt'

feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                'recovery_index', 'response_rate']

SKIP_SECTORS = ['healthcare', 'packaging', 'printing', 'recycling', 'renewable', 'waste_mgmt', 'water']
SKIP_SUBSECTORS = {
    'security': ['aml'],
    'energy': ['well_services', 'oilfield_services'],
    'defense': ['r_and_d'],
    'biotech': ['pathology_bio'],
    'logistics': ['oms'],
    'finance': ['blockchain_fin'],
}

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

class OODDetector(nn.Module):
    def __init__(self, input_dim=10, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=1)

class SectorModel(nn.Module):
    def __init__(self, input_dim=10, n_sectors=43):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_sectors)
        )
    def forward(self, x): return self.net(x)

class SubsectorModel(nn.Module):
    def __init__(self, input_dim=10, n_sectors=43, n_subsectors=50):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, 128)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 128, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_subsectors)
        )
    def forward(self, x, sector): return self.net(torch.cat([x, self.emb(sector)], dim=1))

def load_data():
    log("Loading data...")
    all_data = []
    sector_to_id, id_to_sector, sector_sub_to_id, sector_bases = {}, {}, {}, {}
    current_id = 0
    
    for pf in sorted(DATA_DIR.glob('*.parquet')):
        sector = pf.stem
        if sector in SKIP_SECTORS:
            continue
        
        df = pd.read_parquet(pf)
        if sector in SKIP_SUBSECTORS:
            df = df[~df['subsector'].isin(SKIP_SUBSECTORS[sector])]
        
        sector_to_id[sector] = current_id
        id_to_sector[current_id] = sector
        sector_sub_to_id[current_id] = {s: i for i, s in enumerate(sorted(df['subsector'].unique()))}
        sector_bases[sector] = df[feature_cols].min().min()
        
        df['sector_id'] = current_id
        df['subsector_id'] = df['subsector'].map(sector_sub_to_id[current_id])
        all_data.append(df)
        current_id += 1
    
    combined = pd.concat(all_data, ignore_index=True)
    log(f"Loaded {len(combined)} samples, {len(sector_to_id)} sectors")
    return combined, sector_to_id, id_to_sector, sector_sub_to_id, sector_bases

def normalize_data(df, sector_bases, sector_to_id):
    X = df[feature_cols].values.astype(np.float32)
    for sector, sid in sector_to_id.items():
        X[df['sector_id'].values == sid] -= sector_bases[sector]
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max

def create_missing(X, rate=0.3):
    mask = (np.random.rand(*X.shape) > rate).astype(np.float32)
    return X * mask, mask

def evaluate_model(midas, sector_model, subsector_model, X, y_sec, y_sub, rate=0.3):
    midas.eval(); sector_model.eval(); subsector_model.eval()
    X_miss, mask = create_missing(X, rate)
    with torch.no_grad():
        X_t, m_t = torch.FloatTensor(X_miss), torch.FloatTensor(mask)
        X_imp = midas.impute(X_t, m_t) if rate > 0 else X_t
        pred_s = sector_model(X_imp).argmax(1).numpy()
        pred_sub = subsector_model(X_imp, torch.LongTensor(y_sec)).argmax(1).numpy()
    return (pred_s == y_sec).mean() * 100, (pred_sub == y_sub).mean() * 100

if __name__ == "__main__":
    open(LOG_FILE, 'w').close()
    log("="*70)
    log("SCHEMALABSAI V2 - OOD + MIDAS + Self-Learning + Early Stop")
    log("="*70)
    
    start = time.time()
    
    df, sector_to_id, id_to_sector, sector_sub_to_id, sector_bases = load_data()
    n_sectors = len(sector_to_id)
    
    X_norm, X_min, X_max = normalize_data(df, sector_bases, sector_to_id)
    y_sec, y_sub = df['sector_id'].values, df['subsector_id'].values
    
    np.random.seed(42)
    idx = np.random.permutation(len(X_norm))
    split = int(0.9 * len(idx))
    X_train, X_val = X_norm[idx[:split]], X_norm[idx[split:]]
    y_sec_train, y_sec_val = y_sec[idx[:split]], y_sec[idx[split:]]
    y_sub_train, y_sub_val = y_sub[idx[:split]], y_sub[idx[split:]]
    
    log(f"Train: {len(X_train)}, Val: {len(X_val)}, Sectors: {n_sectors}")
    
    ood = OODDetector()
    midas = MIDAS()
    model_s = SectorModel(n_sectors=n_sectors)
    model_sub = SubsectorModel(n_sectors=n_sectors)
    
    # Phase 0: OOD
    log("\n[Phase 0] OOD Detector")
    opt = AdamW(ood.parameters(), lr=1e-3)
    for ep in range(100):
        ood.train()
        X_t = torch.FloatTensor(X_train)
        opt.zero_grad()
        loss = F.mse_loss(ood(X_t), X_t)
        loss.backward()
        opt.step()
        if (ep+1) % 25 == 0: log(f"  Epoch {ep+1} - Loss: {loss.item():.6f}")
    
    ood.eval()
    with torch.no_grad():
        errs = ood.reconstruction_error(torch.FloatTensor(X_train))
        ood_threshold = errs.mean().item() + 3 * errs.std().item()
    log(f"  OOD Threshold: {ood_threshold:.6f}")
    
    # Phase 1: MIDAS (Early Stop)
    log("\n[Phase 1] MIDAS Training")
    opt = AdamW(midas.parameters(), lr=1e-3)
    best_loss, patience, no_improve = float('inf'), 30, 0
    best_midas = None
    
    for ep in range(500):
        midas.train()
        X_miss, mask = create_missing(X_train, np.random.uniform(0.1, 0.5))
        X_t, X_m, m_t = torch.FloatTensor(X_train), torch.FloatTensor(X_miss), torch.FloatTensor(mask)
        opt.zero_grad()
        pred = midas(X_m, m_t)
        loss = F.mse_loss(pred * (1-m_t), X_t * (1-m_t))
        loss.backward()
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_midas = {k: v.cpu().clone() for k, v in midas.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (ep+1) % 50 == 0: log(f"  Epoch {ep+1} - Loss: {loss.item():.6f} - Best: {best_loss:.6f}")
        if no_improve >= patience:
            log(f"  Early stop at epoch {ep+1}")
            break
    
    midas.load_state_dict(best_midas)
    
    # Phase 2: Classifiers (Early Stop)
    log("\n[Phase 2] Classifier Training")
    opt_s = AdamW(model_s.parameters(), lr=1e-3)
    opt_sub = AdamW(model_sub.parameters(), lr=1e-3)
    best_acc, patience, no_improve = 0, 30, 0
    best_state = None
    
    for ep in range(500):
        model_s.train(); model_sub.train(); midas.eval()
        X_miss, mask = create_missing(X_train, np.random.uniform(0, 0.3))
        
        with torch.no_grad():
            X_imp = midas.impute(torch.FloatTensor(X_miss), torch.FloatTensor(mask))
        
        y_s_t, y_sub_t = torch.LongTensor(y_sec_train), torch.LongTensor(y_sub_train)
        
        opt_s.zero_grad()
        loss_s = F.cross_entropy(model_s(X_imp), y_s_t)
        loss_s.backward()
        opt_s.step()
        
        opt_sub.zero_grad()
        loss_sub = F.cross_entropy(model_sub(X_imp, y_s_t), y_sub_t)
        loss_sub.backward()
        opt_sub.step()
        
        # Validation
        sec_acc, sub_acc = evaluate_model(midas, model_s, model_sub, X_val, y_sec_val, y_sub_val, rate=0.3)
        
        if sub_acc > best_acc:
            best_acc = sub_acc
            best_state = {
                'sector': {k: v.cpu().clone() for k, v in model_s.state_dict().items()},
                'subsector': {k: v.cpu().clone() for k, v in model_sub.state_dict().items()}
            }
            no_improve = 0
        else:
            no_improve += 1
        
        if (ep+1) % 50 == 0: log(f"  Epoch {ep+1} - Val Sub: {sub_acc:.1f}% - Best: {best_acc:.1f}%")
        if no_improve >= patience:
            log(f"  Early stop at epoch {ep+1}")
            break
    
    model_s.load_state_dict(best_state['sector'])
    model_sub.load_state_dict(best_state['subsector'])
    
    # Phase 3: Self-Learning (Early Stop)
    log("\n[Phase 3] Self-Learning")
    opt_s = AdamW(model_s.parameters(), lr=5e-4)
    opt_sub = AdamW(model_sub.parameters(), lr=5e-4)
    best_acc, no_improve = best_acc, 0
    
    for rnd in range(10):
        model_s.eval(); model_sub.eval(); midas.eval()
        
        X_miss, mask = create_missing(X_train, 0.4)
        with torch.no_grad():
            X_imp = midas.impute(torch.FloatTensor(X_miss), torch.FloatTensor(mask))
            probs_s = F.softmax(model_s(X_imp), dim=1)
            conf_s, pseudo_s = probs_s.max(1)
            probs_sub = F.softmax(model_sub(X_imp, pseudo_s), dim=1)
            conf_sub, pseudo_sub = probs_sub.max(1)
        
        high_conf = (conf_s > 0.9) & (conf_sub > 0.9)
        log(f"  Round {rnd+1} - High conf: {high_conf.sum().item()}/{len(X_train)}")
        
        y_s_mix = y_sec_train.copy()
        y_sub_mix = y_sub_train.copy()
        y_s_mix[high_conf.numpy()] = pseudo_s[high_conf].numpy()
        y_sub_mix[high_conf.numpy()] = pseudo_sub[high_conf].numpy()
        
        model_s.train(); model_sub.train()
        for _ in range(50):
            X_miss, mask = create_missing(X_train, np.random.uniform(0.2, 0.5))
            with torch.no_grad():
                X_imp = midas.impute(torch.FloatTensor(X_miss), torch.FloatTensor(mask))
            
            opt_s.zero_grad()
            F.cross_entropy(model_s(X_imp), torch.LongTensor(y_s_mix)).backward()
            opt_s.step()
            
            opt_sub.zero_grad()
            F.cross_entropy(model_sub(X_imp, torch.LongTensor(y_s_mix)), torch.LongTensor(y_sub_mix)).backward()
            opt_sub.step()
        
        sec_acc, sub_acc = evaluate_model(midas, model_s, model_sub, X_val, y_sec_val, y_sub_val, rate=0.3)
        log(f"    Val: Sec {sec_acc:.1f}% - Sub {sub_acc:.1f}%")
        
        if sub_acc > best_acc:
            best_acc = sub_acc
            best_state = {
                'sector': {k: v.cpu().clone() for k, v in model_s.state_dict().items()},
                'subsector': {k: v.cpu().clone() for k, v in model_sub.state_dict().items()}
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                log(f"  Early stop at round {rnd+1}")
                break
    
    model_s.load_state_dict(best_state['sector'])
    model_sub.load_state_dict(best_state['subsector'])
    
    # Final Eval
    log("\n[Final Evaluation]")
    for rate in [0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        sec, sub = evaluate_model(midas, model_s, model_sub, X_val, y_sec_val, y_sub_val, rate)
        log(f"  {int(rate*100)}% missing - Sec: {sec:.1f}% - Sub: {sub:.1f}%")
    
    # Save
    torch.save({
        'midas': midas.state_dict(),
        'ood': ood.state_dict(),
        'ood_threshold': ood_threshold,
        'sector_model': model_s.state_dict(),
        'subsector_model': model_sub.state_dict(),
        'sector_to_id': sector_to_id,
        'id_to_sector': id_to_sector,
        'sector_sub_to_id': sector_sub_to_id,
        'sector_bases': sector_bases,
        'X_min': X_min.tolist(),
        'X_max': X_max.tolist(),
        'n_sectors': n_sectors,
        'feature_cols': feature_cols,
        'version': 'V2_OOD_EarlyStop'
    }, CHECKPOINT_DIR / 'schemalabsai_v2.pt')
    
    log(f"\n{'='*70}")
    log(f"DONE - {(time.time()-start)/60:.1f} min")
    log(f"Best Val Subsector: {best_acc:.1f}%")
    log(f"{'='*70}")
