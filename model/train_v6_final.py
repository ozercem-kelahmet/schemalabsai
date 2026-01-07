#!/usr/bin/env python3
"""
SchemaLabsAI V6 - MIDAS + ProtoNet + Self-Learning
%83 M70 başarılı model
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CKPT_DIR = Path('../checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

def generate_data(nf, nc, ns, seed=None):
    if seed is not None:
        np.random.seed(seed)
    spc = max(1, ns // nc)
    centers = np.random.randn(nc, nf).astype(np.float32) * 5.0
    X, y = [], []
    for c in range(nc):
        n_samples = spc if c < nc - 1 else max(1, ns - len(y))
        if n_samples > 0:
            samples = centers[c] + np.random.randn(n_samples, nf).astype(np.float32) * 0.3
            X.append(samples)
            y.extend([c] * n_samples)
    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y

class MIDASProto(nn.Module):
    def __init__(self, nf, embed_dim=128, n_iter=10):
        super().__init__()
        self.nf, self.n_iter, self.embed_dim = nf, n_iter, embed_dim
        h = nf * 3
        self.midas = nn.Sequential(
            nn.Linear(nf * 2, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, nf)
        )
        self.proto = nn.Sequential(
            nn.Linear(nf, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim)
        )
    
    def impute_step(self, x, m):
        return self.midas(torch.cat([x, m], dim=-1))
    
    def impute(self, x, m):
        cur = x * m
        for _ in range(self.n_iter):
            cur = x * m + self.impute_step(cur, m) * (1 - m)
        return cur
    
    def get_embeddings(self, x, m=None):
        if m is not None:
            x = self.impute(x, m)
        return self.proto(x)
    
    def predict(self, sx, sy, qx, nc, sm=None, qm=None):
        s_emb = self.get_embeddings(sx, sm)
        q_emb = self.get_embeddings(qx, qm)
        
        prototypes = []
        for c in range(nc):
            mask = sy == c
            proto = s_emb[mask].mean(0) if mask.sum() > 0 else torch.zeros(self.embed_dim, device=sx.device)
            prototypes.append(proto)
        
        return -torch.cdist(q_emb, torch.stack(prototypes))
    
    def self_learning(self, sx, sy, qx, nc, sm=None, qm=None, n_rounds=5, threshold=0.9):
        """Self-Learning: pseudo-labeling ile iteratif iyileştirme"""
        s_emb = self.get_embeddings(sx, sm)
        q_emb = self.get_embeddings(qx, qm)
        
        cur_s_emb = s_emb.clone()
        cur_sy = sy.clone()
        rem_q_emb = q_emb.clone()
        rem_idx = torch.arange(len(qx), device=qx.device)
        
        for r in range(n_rounds):
            if len(rem_q_emb) == 0:
                break
            
            # Prototype hesapla
            prototypes = []
            for c in range(nc):
                mask = cur_sy == c
                proto = cur_s_emb[mask].mean(0) if mask.sum() > 0 else torch.zeros(self.embed_dim, device=sx.device)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes)
            
            # Predict
            logits = -torch.cdist(rem_q_emb, prototypes)
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)
            
            # Confident olanları seç
            confident = conf >= threshold
            if confident.sum() == 0:
                threshold -= 0.05
                if threshold < 0.5:
                    break
                continue
            
            # Support'a ekle
            cur_s_emb = torch.cat([cur_s_emb, rem_q_emb[confident]])
            cur_sy = torch.cat([cur_sy, preds[confident]])
            
            # Kalan query'ler
            rem_q_emb = rem_q_emb[~confident]
            rem_idx = rem_idx[~confident]
        
        # Final prediction
        prototypes = []
        for c in range(nc):
            mask = cur_sy == c
            proto = cur_s_emb[mask].mean(0) if mask.sum() > 0 else torch.zeros(self.embed_dim, device=sx.device)
            prototypes.append(proto)
        
        return -torch.cdist(q_emb, torch.stack(prototypes))

def train():
    nf = 100
    dev = torch.device('cpu')
    
    print("=" * 70)
    print("MIDAS + PROTONET + SELF-LEARNING V6")
    print("=" * 70)
    
    model = MIDASProto(nf, embed_dim=128, n_iter=10).to(dev)
    opt = AdamW(model.parameters(), lr=1e-3)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 1: MIDAS (1000 ep)
    print("\nPhase 1: MIDAS (1000 ep)...")
    for ep in range(1000):
        model.train()
        nc = np.random.choice([100, 500, 1000])
        X, _ = generate_data(nf, nc, min(2000, nc*5), seed=np.random.randint(0, 100000))
        xb = torch.tensor(X, device=dev)
        mr = 0.5 + np.random.rand() * 0.3
        m = (torch.rand_like(xb) > mr).float()
        pred = model.impute_step(xb * m, m)
        loss = (((pred - xb)**2) * (1 - m)).sum() / ((1 - m).sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep+1) % 250 == 0:
            print(f"  Ep {ep+1}: loss={loss.item():.4f}")
    
    # Phase 2: ProtoNet (1000 ep)
    print("\nPhase 2: ProtoNet (1000 ep)...")
    for ep in range(1000):
        model.train()
        nc = np.random.choice([50, 100, 200, 500, 1000])
        spc = 5
        ns = nc * spc * 2
        
        X, y = generate_data(nf, nc, ns, seed=np.random.randint(0, 100000))
        
        n_support = nc * spc
        sx = torch.tensor(X[:n_support], device=dev)
        sy = torch.tensor(y[:n_support], device=dev)
        qx = torch.tensor(X[n_support:], device=dev)
        qy = torch.tensor(y[n_support:], device=dev)
        
        mr = 0.5 + np.random.rand() * 0.3
        sm = (torch.rand_like(sx) > mr).float()
        qm = (torch.rand_like(qx) > mr).float()
        
        logits = model.predict(sx, sy, qx, nc, sm, qm)
        loss = nn.CrossEntropyLoss()(logits, qy)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (ep+1) % 200 == 0:
            model.eval()
            X_te, y_te = generate_data(nf, 100, 1000, seed=999)
            tsx = torch.tensor(X_te[:500], device=dev)
            tsy = torch.tensor(y_te[:500], device=dev)
            tqx = torch.tensor(X_te[500:], device=dev)
            tqy = torch.tensor(y_te[500:], device=dev)
            torch.manual_seed(999)
            tsm = (torch.rand_like(tsx) > 0.7).float()
            tqm = (torch.rand_like(tqx) > 0.7).float()
            with torch.no_grad():
                logits = model.predict(tsx, tsy, tqx, 100, tsm, tqm)
                acc = (logits.argmax(-1) == tqy).float().mean().item()
            print(f"  Ep {ep+1}: 100c M70={acc*100:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), CKPT_DIR / 'v6_midasproto.pt')
    print(f"\n✅ Model saved: {CKPT_DIR / 'v6_midasproto.pt'}")
    
    return model

def test(model):
    nf = 100
    dev = torch.device('cpu')
    
    print("\n" + "=" * 70)
    print("FULL TEST")
    print("=" * 70)
    
    model.eval()
    
    print(f"\n{'NC':<6} | {'M0':<7} | {'M30':<7} | {'M50':<7} | {'M70':<7} | {'M70+SL':<7}")
    print("-" * 55)
    
    for test_nc in [100, 500, 1000]:
        results = {0: [], 0.3: [], 0.5: [], 0.7: [], 'sl': []}
        
        for seed in [100, 200, 300]:
            spc = 5
            n_support = test_nc * spc
            n_query = min(test_nc * 2, 2000)
            
            X, y = generate_data(nf, test_nc, n_support + n_query, seed=seed)
            
            sx = torch.tensor(X[:n_support], device=dev)
            sy = torch.tensor(y[:n_support], device=dev)
            qx = torch.tensor(X[n_support:n_support+n_query], device=dev)
            qy = torch.tensor(y[n_support:n_support+n_query], device=dev)
            
            for mr in [0, 0.3, 0.5, 0.7]:
                torch.manual_seed(seed)
                sm = (torch.rand_like(sx) > mr).float() if mr > 0 else None
                qm = (torch.rand_like(qx) > mr).float() if mr > 0 else None
                
                with torch.no_grad():
                    logits = model.predict(sx, sy, qx, test_nc, sm, qm)
                    acc = (logits.argmax(-1) == qy).float().mean().item()
                results[mr].append(acc)
            
            # M70 + Self-Learning
            torch.manual_seed(seed)
            sm = (torch.rand_like(sx) > 0.7).float()
            qm = (torch.rand_like(qx) > 0.7).float()
            with torch.no_grad():
                logits = model.self_learning(sx, sy, qx, test_nc, sm, qm)
                acc = (logits.argmax(-1) == qy).float().mean().item()
            results['sl'].append(acc)
        
        m0 = np.mean(results[0]) * 100
        m30 = np.mean(results[0.3]) * 100
        m50 = np.mean(results[0.5]) * 100
        m70 = np.mean(results[0.7]) * 100
        sl = np.mean(results['sl']) * 100
        
        status = "✅" if m70 >= 70 else ""
        print(f"{test_nc:<6} | {m0:>5.1f}% | {m30:>5.1f}% | {m50:>5.1f}% | {m70:>5.1f}% | {sl:>5.1f}% {status}")
    
    print("=" * 70)

if __name__ == '__main__':
    model = train()
    test(model)
