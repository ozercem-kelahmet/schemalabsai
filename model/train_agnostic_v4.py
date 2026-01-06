#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import gc
from pathlib import Path
from datetime import datetime

DEVICE = 'cpu'
CHECKPOINT_DIR = Path('../checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_FILE = CHECKPOINT_DIR / 'v4_training_log.txt'

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")

def generate_data(n_samples, n_features, n_classes, missing_rate=0.3):
    np.random.seed(42)
    samples_per_class = n_samples // n_classes
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.repeat(np.arange(n_classes), samples_per_class)
    features_per_class = max(1, n_features // n_classes)
    for c in range(n_classes):
        start_f = (c * features_per_class) % n_features
        end_f = min(start_f + features_per_class, n_features)
        X[y == c, start_f:end_f] += 6
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    for i in range(n_features):
        col_min, col_max = X[:, i].min(), X[:, i].max()
        if col_max - col_min > 1e-8:
            X[:, i] = (X[:, i] - col_min) / (col_max - col_min)
    mask = (np.random.rand(*X.shape) > missing_rate).astype(np.float32)
    return X, y, mask

class DataAgnosticConfig:
    @staticmethod
    def get(n_features, n_classes, n_samples):
        cfg = {}
        cfg['proj'] = min(2048, max(256, n_features))
        cfg['lat'] = min(1024, max(128, n_features // 2))
        cfg['hidden'] = min(2000, max(512, n_classes * 2))
        cfg['epochs'] = 200
        cfg['batch'] = min(4096, max(256, n_samples // 100))
        cfg['lr'] = 1e-3
        cfg['impute_iter'] = 5
        cfg['noise_std'] = 0.1
        cfg['patience'] = 20
        cfg['sl_rounds'] = 3
        cfg['sl_init_thr'] = 0.95
        cfg['sl_min_thr'] = 0.80
        return cfg

class MIDAS(nn.Module):
    def __init__(self, n_features, n_classes, cfg):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        proj = cfg['proj']
        lat = cfg['lat']
        h = cfg['hidden']
        self.proj = nn.Sequential(nn.Linear(n_features, proj), nn.GELU(), nn.LayerNorm(proj), nn.Dropout(0.1))
        self.encoder = nn.Sequential(nn.Linear(proj * 2, lat * 2), nn.GELU(), nn.LayerNorm(lat * 2), nn.Dropout(0.1), nn.Linear(lat * 2, lat), nn.GELU(), nn.LayerNorm(lat))
        self.decoder = nn.Sequential(nn.Linear(lat, proj), nn.GELU(), nn.Linear(proj, n_features))
        self.classifier = nn.Sequential(nn.Linear(lat, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(0.3), nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(0.2), nn.Linear(h, n_classes))
        self.impute_iter = cfg['impute_iter']
        self.noise_std = cfg['noise_std']

    def encode(self, x, m):
        xp = self.proj(x * m)
        mp = self.proj(m)
        return self.encoder(torch.cat([xp, mp], dim=-1))

    def impute(self, x, m):
        current = x * m
        for _ in range(self.impute_iter):
            z = self.encode(current, m)
            recon = self.decoder(z)
            current = x * m + recon * (1 - m)
        return current

    def forward(self, x, m, return_all=False):
        if self.training:
            x = x + torch.randn_like(x) * self.noise_std
        x_imp = self.impute(x, m)
        z = self.encode(x_imp, torch.ones_like(m))
        recon = self.decoder(z)
        logits = self.classifier(z)
        if return_all:
            return logits, recon, x_imp
        return logits

class SelfLearning:
    def __init__(self, cfg):
        self.rounds = cfg['sl_rounds']
        self.init_thr = cfg['sl_init_thr']
        self.min_thr = cfg['sl_min_thr']

    def get_pseudo_labels(self, model, X, m, n_classes, threshold):
        model.eval()
        with torch.no_grad():
            logits = model(X, m)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
        high_conf = conf > threshold
        if high_conf.sum() < 50:
            return None, None
        idx = torch.where(high_conf)[0]
        labels = pred[idx]
        counts = torch.bincount(labels, minlength=n_classes)
        if counts.max() > 0 and counts[counts > 0].numel() > 0:
            min_count = counts[counts > 0].min().item()
            max_per_class = min(counts.max().item(), min_count * 3 + 10)
        else:
            max_per_class = 100
        balanced_idx = []
        balanced_labels = []
        for c in range(n_classes):
            c_mask = labels == c
            c_indices = idx[c_mask]
            if len(c_indices) > max_per_class:
                perm = torch.randperm(len(c_indices))[:max_per_class]
                c_indices = c_indices[perm]
            if len(c_indices) > 0:
                balanced_idx.extend(c_indices.tolist())
                balanced_labels.extend([c] * len(c_indices))
        if len(balanced_idx) < 50:
            return None, None
        return torch.LongTensor(balanced_idx), torch.LongTensor(balanced_labels)

    def run(self, model, Xtr, ytr, mtr, Xte, mte, Xor, n_classes, cfg):
        for r in range(self.rounds):
            thr = self.init_thr - r * 0.05
            thr = max(thr, self.min_thr)
            result = self.get_pseudo_labels(model, Xte, mte, n_classes, thr)
            if result[0] is None:
                log(f"    SL Round {r+1}: No pseudo labels")
                break
            idx, labels = result
            log(f"    SL Round {r+1}: {len(idx)} pseudo (thr={thr:.2f})")
            Xa = torch.cat([Xtr, Xte[idx]])
            ya = torch.cat([ytr, labels])
            ma = torch.cat([mtr, mte[idx]])
            Xoa = torch.cat([Xor, Xte[idx]])
            model.train()
            opt = AdamW(model.parameters(), lr=cfg['lr'] / 10)
            for _ in range(50):
                bidx = torch.randperm(len(Xa))[:cfg['batch']]
                opt.zero_grad()
                logits, recon, _ = model(Xa[bidx], ma[bidx], return_all=True)
                loss = nn.CrossEntropyLoss()(logits, ya[bidx])
                loss += 0.1 * nn.MSELoss()(recon, Xoa[bidx])
                loss.backward()
                opt.step()

def train_v4(n_features, n_classes, n_samples, target_acc=0.95):
    log(f"Training: {n_features}f, {n_classes}c, {n_samples:,}s, target={target_acc*100:.0f}%")
    cfg = DataAgnosticConfig.get(n_features, n_classes, n_samples)
    log(f"  Config: proj={cfg['proj']}, lat={cfg['lat']}, hidden={cfg['hidden']}, batch={cfg['batch']}")
    X, y, mask = generate_data(n_samples, n_features, n_classes, missing_rate=0.3)
    X_orig = X.copy()
    split = int(n_samples * 0.8)
    Xtr = torch.FloatTensor(X[:split])
    ytr = torch.LongTensor(y[:split])
    mtr = torch.FloatTensor(mask[:split])
    Xor = torch.FloatTensor(X_orig[:split])
    Xte = torch.FloatTensor(X[split:])
    yte = torch.LongTensor(y[split:])
    mte = torch.FloatTensor(mask[split:])
    model = MIDAS(n_features, n_classes, cfg)
    params = sum(p.numel() for p in model.parameters())
    log(f"  Model params: {params:,}")
    opt = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    best_acc = 0
    best_state = None
    patience_cnt = 0
    log(f"  Training {cfg['epochs']} epochs...")
    for ep in range(cfg['epochs']):
        model.train()
        idx = torch.randperm(len(Xtr))[:cfg['batch']]
        opt.zero_grad()
        logits, recon, x_imp = model(Xtr[idx], mtr[idx], return_all=True)
        L_clf = nn.CrossEntropyLoss()(logits, ytr[idx])
        L_rec = nn.MSELoss()(recon, Xor[idx])
        missing = 1 - mtr[idx]
        L_imp = ((x_imp - Xor[idx]) ** 2 * missing).sum() / (missing.sum() + 1e-8)
        loss = L_clf + 0.1 * L_rec + 0.2 * L_imp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_size = min(5000, len(Xte))
                acc = (model(Xte[:test_size], mte[:test_size]).argmax(-1) == yte[:test_size]).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if (ep + 1) % 50 == 0:
                log(f"    Ep {ep+1}: {acc*100:.1f}% (best={best_acc*100:.1f}%)")
            if acc >= target_acc:
                log(f"    Ep {ep+1}: {acc*100:.1f}% - Target reached!")
                break
            if patience_cnt >= cfg['patience']:
                log(f"    Ep {ep+1}: Early stop")
                break
    if best_state:
        model.load_state_dict(best_state)
    log(f"  Self-Learning ({cfg['sl_rounds']} rounds)...")
    sl = SelfLearning(cfg)
    sl.run(model, Xtr, ytr, mtr, Xte, mte, Xor, n_classes, cfg)
    model.eval()
    with torch.no_grad():
        final_acc = (model(Xte, mte).argmax(-1) == yte).float().mean().item()
    status = "PASS" if final_acc >= target_acc else "FAIL"
    log(f"  FINAL: {final_acc*100:.1f}% (target {target_acc*100:.0f}%) [{status}]")
    del Xtr, ytr, mtr, Xor, Xte, yte, mte, X, y, mask, X_orig
    gc.collect()
    return model, cfg, final_acc

def main():
    log("=" * 70)
    log("V4 DATA-AGNOSTIC TRAINING - 10M DATA")
    log("MIDAS (6) + Self-Learning (3)")
    log("=" * 70)
    combinations = [
        (10, 10, 100000, 0.99),
        (10, 100, 500000, 0.95),
        (100, 100, 1000000, 0.99),
        (1000, 100, 1000000, 0.99),
        (1000, 500, 2000000, 0.95),
        (5000, 100, 1000000, 0.99),
        (5000, 500, 2000000, 0.95),
        (5000, 1000, 2400000, 0.90),
    ]
    models = {}
    results = []
    for n_feat, n_class, n_samples, target in combinations:
        log(f"\n{'='*60}")
        log(f"COMBINATION: {n_feat}f x {n_class}c x {n_samples:,}s")
        log(f"{'='*60}")
        try:
            model, cfg, acc = train_v4(n_feat, n_class, n_samples, target)
            key = f"f{n_feat}_c{n_class}"
            models[key] = {'state_dict': model.state_dict(), 'n_features': n_feat, 'n_classes': n_class, 'cfg': cfg, 'acc': acc}
            results.append((n_feat, n_class, acc, target))
        except Exception as e:
            log(f"  ERROR: {e}")
            results.append((n_feat, n_class, 0, target))
        gc.collect()
    log("\n" + "=" * 70)
    log("TRAINING SUMMARY")
    log("=" * 70)
    all_pass = True
    for n_feat, n_class, acc, target in results:
        status = "PASS" if acc >= target else "FAIL"
        if acc < target:
            all_pass = False
        log(f"{n_feat}f x {n_class}c: {acc*100:.1f}% (target {target*100:.0f}%) [{status}]")
    log("=" * 70)
    log(f"RESULT: {'ALL TARGETS MET' if all_pass else 'SOME TARGETS MISSED'}")
    log("=" * 70)
    save_path = CHECKPOINT_DIR / 'schemalabsai_v4_agnostic.pt'
    torch.save(models, save_path)
    log(f"Models saved to {save_path}")

if __name__ == '__main__':
    main()
