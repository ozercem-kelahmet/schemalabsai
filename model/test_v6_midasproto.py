#!/usr/bin/env python3
"""
SchemaLabsAI V6 - MIDAS + ProtoNet + Self-Learning
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from pathlib import Path
import gc

DATA_DIR = Path('../data_agnostic')
CKPT_DIR = Path('../checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

class MIDASProto(nn.Module):
    def __init__(self, max_nf=1000, embed_dim=128, n_iter=10):
        super().__init__()
        self.max_nf = max_nf
        self.n_iter = n_iter
        self.embed_dim = embed_dim
        
        h = max_nf * 3
        self.midas = nn.Sequential(
            nn.Linear(max_nf * 2, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, max_nf)
        )
        self.proto = nn.Sequential(
            nn.Linear(max_nf, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim)
        )
    
    def pad(self, x, nf):
        if nf == self.max_nf:
            return x, torch.ones_like(x)
        batch = x.shape[0]
        x_pad = torch.zeros(batch, self.max_nf, device=x.device, dtype=x.dtype)
        x_pad[:, :nf] = x
        m = torch.zeros(batch, self.max_nf, device=x.device)
        m[:, :nf] = 1.0
        return x_pad, m
    
    def impute_step(self, x, m):
        return self.midas(torch.cat([x, m], dim=-1))
    
    def impute(self, x, m):
        cur = x * m
        for _ in range(self.n_iter):
            pred = self.impute_step(cur, m)
            cur = x * m + pred * (1 - m)
        return cur
    
    def forward_with_pad(self, x, nf, user_mask=None):
        x_pad, pad_mask = self.pad(x, nf)
        if user_mask is not None:
            full_mask = torch.zeros_like(x_pad)
            full_mask[:, :nf] = user_mask
        else:
            full_mask = pad_mask
        x_imp = self.impute(x_pad, full_mask)
        return self.proto(x_imp)
    
    def predict(self, support_x, support_y, query_x, nc, nf, support_m=None, query_m=None):
        s_emb = self.forward_with_pad(support_x, nf, support_m)
        q_emb = self.forward_with_pad(query_x, nf, query_m)
        
        prototypes = []
        for c in range(nc):
            mask = support_y == c
            if mask.sum() > 0:
                prototypes.append(s_emb[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(self.embed_dim, device=support_x.device))
        prototypes = torch.stack(prototypes)
        
        return -torch.cdist(q_emb, prototypes)
    
    def self_learning(self, support_x, support_y, query_x, nc, nf, 
                      support_m=None, query_m=None, n_rounds=5, threshold=0.95):
        s_emb = self.forward_with_pad(support_x, nf, support_m)
        q_emb = self.forward_with_pad(query_x, nf, query_m)
        
        current_s_emb, current_sy = s_emb.clone(), support_y.clone()
        remaining_q_emb = q_emb.clone()
        
        for _ in range(n_rounds):
            if len(remaining_q_emb) == 0:
                break
            
            prototypes = []
            for c in range(nc):
                mask = current_sy == c
                if mask.sum() > 0:
                    prototypes.append(current_s_emb[mask].mean(dim=0))
                else:
                    prototypes.append(torch.zeros(self.embed_dim, device=s_emb.device))
            prototypes = torch.stack(prototypes)
            
            logits = -torch.cdist(remaining_q_emb, prototypes)
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)
            
            confident = max_probs >= threshold
            if confident.sum() == 0:
                break
            
            current_s_emb = torch.cat([current_s_emb, remaining_q_emb[confident]], dim=0)
            current_sy = torch.cat([current_sy, preds[confident]], dim=0)
            remaining_q_emb = remaining_q_emb[~confident]
        
        prototypes = []
        for c in range(nc):
            mask = current_sy == c
            if mask.sum() > 0:
                prototypes.append(current_s_emb[mask].mean(dim=0))
            else:
                prototypes.append(torch.zeros(self.embed_dim, device=s_emb.device))
        prototypes = torch.stack(prototypes)
        
        return -torch.cdist(q_emb, prototypes)

def load_parquet_sample(filepath, max_samples=50000):
    df = pd.read_parquet(filepath)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    return X, y

def test_model(model, X, y, nf, nc, dev):
    """Tek dataset test"""
    n_support = min(nc * 5, len(X) // 2)
    n_query = min(500, len(X) - n_support)
    
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    
    sx = torch.tensor(X[perm[:n_support]], device=dev)
    sy = torch.tensor(y[perm[:n_support]], device=dev)
    qx = torch.tensor(X[perm[n_support:n_support+n_query]], device=dev)
    qy = torch.tensor(y[perm[n_support:n_support+n_query]], device=dev)
    
    results = {}
    for mr in [0, 0.3, 0.5, 0.7]:
        torch.manual_seed(42)
        sm = (torch.rand_like(sx) > mr).float() if mr > 0 else None
        qm = (torch.rand_like(qx) > mr).float() if mr > 0 else None
        
        with torch.no_grad():
            logits = model.predict(sx, sy, qx, nc, nf, sm, qm)
            results[mr] = (logits.argmax(-1) == qy).float().mean().item()
    
    # M70+SL
    torch.manual_seed(42)
    sm = (torch.rand_like(sx) > 0.7).float()
    qm = (torch.rand_like(qx) > 0.7).float()
    with torch.no_grad():
        logits = model.self_learning(sx, sy, qx, nc, nf, sm, qm)
        results['sl'] = (logits.argmax(-1) == qy).float().mean().item()
    
    return results

def main():
    print("=" * 70)
    print("SchemaLabsAI V6 - MIDAS + ProtoNet + Self-Learning")
    print("=" * 70)
    
    parquets = sorted(DATA_DIR.glob('prod_*.parquet'))
    print(f"Found {len(parquets)} parquet files\n")
    
    datasets = {}
    max_nf = 0
    for pq in parquets:
        X, y = load_parquet_sample(pq, max_samples=30000)
        nf, nc = X.shape[1], len(np.unique(y))
        datasets[pq.name] = {'X': X, 'y': y, 'nf': nf, 'nc': nc}
        max_nf = max(max_nf, nf)
        print(f"  {pq.name}: {nf}f, {nc}c, {len(X):,}s")
    
    dev = torch.device('cpu')
    model = MIDASProto(max_nf=max_nf, embed_dim=128, n_iter=10).to(dev)
    opt = AdamW(model.parameters(), lr=1e-3)
    
    print(f"\nMax features: {max_nf}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    dataset_names = list(datasets.keys())
    
    # Phase 1: MIDAS
    print("\n" + "-" * 70)
    print("Phase 1: MIDAS Training (300 ep)")
    print("-" * 70)
    
    for ep in range(300):
        model.train()
        ds = datasets[np.random.choice(dataset_names)]
        X, nf = ds['X'], ds['nf']
        
        idx = np.random.choice(len(X), min(256, len(X)), replace=False)
        xb = torch.tensor(X[idx], device=dev)
        xb_pad, pad_mask = model.pad(xb, nf)
        
        mr = 0.3 + np.random.rand() * 0.4
        user_mask = (torch.rand(xb.shape[0], nf, device=dev) > mr).float()
        full_mask = torch.zeros_like(xb_pad)
        full_mask[:, :nf] = user_mask
        
        pred = model.impute_step(xb_pad * full_mask, full_mask)
        target_mask = pad_mask * (1 - full_mask)
        loss = (((pred - xb_pad)**2) * target_mask).sum() / (target_mask.sum() + 1e-8)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (ep + 1) % 100 == 0:
            print(f"  Ep {ep+1}: loss={loss.item():.4f}")
    
    # Phase 2: ProtoNet
    print("\n" + "-" * 70)
    print("Phase 2: ProtoNet Training (300 ep)")
    print("-" * 70)
    
    for ep in range(300):
        model.train()
        ds = datasets[np.random.choice(dataset_names)]
        X, y, nf, nc = ds['X'], ds['y'], ds['nf'], ds['nc']
        
        n_support = min(nc * 5, len(X) // 2)
        perm = np.random.permutation(len(X))
        
        sx = torch.tensor(X[perm[:n_support]], device=dev)
        sy = torch.tensor(y[perm[:n_support]], device=dev)
        qx = torch.tensor(X[perm[n_support:n_support+256]], device=dev)
        qy = torch.tensor(y[perm[n_support:n_support+256]], device=dev)
        
        mr = np.random.rand() * 0.7
        sm = (torch.rand_like(sx) > mr).float() if mr > 0 else None
        qm = (torch.rand_like(qx) > mr).float() if mr > 0 else None
        
        logits = model.predict(sx, sy, qx, nc, nf, sm, qm)
        loss = nn.CrossEntropyLoss()(logits, qy)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (ep + 1) % 100 == 0:
            # Quick test
            model.eval()
            ds_test = datasets[dataset_names[0]]
            res = test_model(model, ds_test['X'], ds_test['y'], ds_test['nf'], ds_test['nc'], dev)
            print(f"  Ep {ep+1}: loss={loss.item():.4f} | M0={res[0]*100:.1f}% M50={res[0.5]*100:.1f}% M70={res[0.7]*100:.1f}%")
            model.train()
    
    # Save
    torch.save(model.state_dict(), CKPT_DIR / 'v6_midasproto.pt')
    print(f"\n✅ Model saved: v6_midasproto.pt")
    
    # Final Test
    print("\n" + "=" * 70)
    print("FINAL TEST")
    print("=" * 70)
    
    model.eval()
    
    print(f"\n{'Dataset':<25} | {'M0':<7} | {'M30':<7} | {'M50':<7} | {'M70':<7} | {'M70+SL':<7}")
    print("-" * 75)
    
    for ds_name, ds in datasets.items():
        res = test_model(model, ds['X'], ds['y'], ds['nf'], ds['nc'], dev)
        print(f"{ds_name:<25} | {res[0]*100:>5.1f}% | {res[0.3]*100:>5.1f}% | {res[0.5]*100:>5.1f}% | {res[0.7]*100:>5.1f}% | {res['sl']*100:>5.1f}%")
    
    print("=" * 70)
    print("✅ COMPLETE")

if __name__ == '__main__':
    main()
