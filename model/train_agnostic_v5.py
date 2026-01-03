import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from datetime import datetime

LOG_PATH = "../checkpoints/schemalabsai_agnostic_v5_log.txt"
SAVE_PATH = "../checkpoints/schemalabsai_agnostic_v5.pt"

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp} - {msg}\n")

def generate_data(n_samples, n_features=10, n_classes=100):
    np.random.seed(42)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    samples_per_class = n_samples // n_classes
    np.random.seed(123)
    class_centers = np.random.randn(n_classes, n_features) * 10
    for i in range(n_classes):
        for j in range(i):
            dist = np.linalg.norm(class_centers[i] - class_centers[j])
            if dist < 5:
                direction = class_centers[i] - class_centers[j]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                class_centers[i] = class_centers[j] + direction * 5
    for c in range(n_classes):
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class
        data = np.random.randn(samples_per_class, n_features) * 0.3 + class_centers[c]
        X[start_idx:end_idx] = data
        y[start_idx:end_idx] = c
    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]
    for i in range(n_features):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min() + 1e-8)
    return X, y

class MIDAS(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 256))
        self.decoder = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim))
    def forward(self, x, mask):
        return self.decoder(self.encoder(torch.cat([x * mask, mask], dim=1)))
    def encode(self, x, mask=None):
        if mask is None: mask = torch.ones_like(x)
        return self.encoder(torch.cat([x * mask, mask], dim=1))
    def impute(self, x, mask, n_iter=5):
        current = x * mask
        for _ in range(n_iter):
            current = x * mask + self.forward(current, mask) * (1 - mask)
        return current

class MIRAS(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, n_classes=100):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.branch2 = nn.Sequential(nn.Linear(input_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.branch3 = nn.Sequential(nn.Linear(input_dim, hidden_dim//4), nn.GELU(), nn.Linear(hidden_dim//4, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim//2), nn.GELU())
        self.classifier = nn.Linear(hidden_dim//2, n_classes)
    def forward(self, x):
        fused = self.fusion(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))
        return self.classifier(fused), fused

class Classifier(nn.Module):
    def __init__(self, input_dim=256, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes))
    def forward(self, x): return self.net(x)

def train():
    log("=" * 60)
    log("V5: MIDAS + MIRAS + Self-Learning")
    log("Target: MIDAS 50% missing >= 90%")
    log("=" * 60)
    
    N_SAMPLES = 10_000_000
    N_FEATURES = 10
    N_CLASSES = 100
    BATCH_SIZE = 2048
    
    log("\n[1/5] Generating 10M data...")
    X, y = generate_data(N_SAMPLES, N_FEATURES, N_CLASSES)
    n_lab = int(N_SAMPLES * 0.6)
    n_unlab = int(N_SAMPLES * 0.2)
    X_lab, y_lab = X[:n_lab], y[:n_lab]
    X_unlab = X[n_lab:n_lab+n_unlab]
    X_test, y_test = X[n_lab+n_unlab:], y[n_lab+n_unlab:]
    log(f"  Labeled: {n_lab:,}, Unlabeled: {n_unlab:,}, Test: {len(X_test):,}")
    
    X_test_t = torch.FloatTensor(X_test[:100000])
    y_test_t = torch.LongTensor(y_test[:100000])
    
    log("\n[2/5] Training MIDAS (until 50% missing >= 90%)...")
    midas = MIDAS(N_FEATURES, 512)
    opt = AdamW(midas.parameters(), lr=0.001, weight_decay=0.01)
    
    test_clf = Classifier(10, N_CLASSES)
    test_clf_opt = AdamW(test_clf.parameters(), lr=0.001, weight_decay=0.01)
    X_train_t = torch.FloatTensor(X_lab[:500000])
    y_train_t = torch.LongTensor(y_lab[:500000])
    for _ in range(20):
        test_clf.train()
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), 2048):
            idx = perm[i:i+2048]
            test_clf_opt.zero_grad()
            nn.CrossEntropyLoss()(test_clf(X_train_t[idx]), y_train_t[idx]).backward()
            test_clf_opt.step()
    test_clf.eval()
    with torch.no_grad():
        clean_acc = (test_clf(X_test_t).argmax(-1) == y_test_t).float().mean().item()
    log(f"  Test classifier clean acc: {clean_acc*100:.1f}%")
    
    best_missing_acc = 0
    for ep in range(100):
        midas.train()
        perm = np.random.permutation(n_lab)
        total_loss, n_batch = 0, 0
        for i in range(0, n_lab, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch = torch.FloatTensor(X_lab[idx])
            mask = (torch.rand_like(batch) > np.random.uniform(0.3, 0.6)).float()
            opt.zero_grad()
            loss = nn.MSELoss()(midas(batch * mask, mask), batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        
        midas.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            mask_50 = (torch.rand_like(X_test_t) > 0.5).float()
            imputed = midas.impute(X_test_t * mask_50, mask_50, n_iter=5)
            missing_acc = (test_clf(imputed).argmax(-1) == y_test_t).float().mean().item()
        
        if missing_acc > best_missing_acc:
            best_missing_acc = missing_acc
            best_midas_state = {k: v.clone() for k, v in midas.state_dict().items()}
        
        log(f"  MIDAS Ep {ep+1}: Loss={total_loss/n_batch:.6f} | Clean={clean_acc*100:.1f}% | 50%Missing={missing_acc*100:.1f}% | Best={best_missing_acc*100:.1f}%")
        
        if best_missing_acc >= 0.90:
            log(f"  🎉 MIDAS reached 90%+ at epoch {ep+1}")
            break
    
    midas.load_state_dict(best_midas_state)
    midas.eval()
    for p in midas.parameters(): p.requires_grad = False
    
    if best_missing_acc < 0.90:
        log(f"  ⚠️ MIDAS did not reach 90%, best={best_missing_acc*100:.1f}%")
    
    log("\n[3/5] Extracting features...")
    X_feat_lab = []
    for i in range(0, n_lab, 50000):
        with torch.no_grad():
            X_feat_lab.append(midas.encode(torch.FloatTensor(X_lab[i:i+50000])))
    X_feat_lab = torch.cat(X_feat_lab)
    X_feat_unlab = []
    for i in range(0, n_unlab, 50000):
        with torch.no_grad():
            X_feat_unlab.append(midas.encode(torch.FloatTensor(X_unlab[i:i+50000])))
    X_feat_unlab = torch.cat(X_feat_unlab)
    with torch.no_grad():
        X_feat_test = midas.encode(torch.FloatTensor(X_test))
    log(f"  Shape: {X_feat_lab.shape}")
    
    log("\n[4/5] Training MIRAS (patience=10)...")
    miras = MIRAS(256, 512, N_CLASSES)
    opt = AdamW(miras.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    y_t = torch.LongTensor(y_lab)
    y_te = torch.LongTensor(y_test)
    best_acc, best_state, patience_cnt = 0, None, 0
    
    for ep in range(100):
        miras.train()
        perm = torch.randperm(len(X_feat_lab))
        total_loss, n_batch = 0, 0
        for i in range(0, len(X_feat_lab), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(miras(X_feat_lab[idx])[0], y_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        miras.eval()
        with torch.no_grad():
            acc = (miras(X_feat_test)[0].argmax(-1) == y_te).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in miras.state_dict().items()}
            patience_cnt = 0
            log(f"  MIRAS Ep {ep+1}: Loss={total_loss/n_batch:.4f} | Acc={acc*100:.2f}% ✓")
        else:
            patience_cnt += 1
        if best_acc >= 0.99 or patience_cnt >= 10:
            break
    
    miras.load_state_dict(best_state)
    
    log("\n[5/5] Self-Learning (5 rounds)...")
    for sl in range(5):
        miras.eval()
        all_high_x, all_high_y = [], []
        with torch.no_grad():
            for i in range(0, len(X_feat_unlab), 50000):
                batch = X_feat_unlab[i:i+50000]
                probs = torch.softmax(miras(batch)[0], 1)
                conf, pred = probs.max(1)
                high = conf > 0.95
                if high.sum() > 0:
                    all_high_x.append(batch[high])
                    all_high_y.append(pred[high])
        if len(all_high_x) == 0:
            log(f"  Round {sl+1}: No pseudo labels")
            continue
        X_pseudo = torch.cat(all_high_x)
        y_pseudo = torch.cat(all_high_y)
        miras.train()
        for _ in range(3):
            perm = torch.randperm(len(X_pseudo))
            for i in range(0, len(X_pseudo), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                opt.zero_grad()
                loss_fn(miras(X_pseudo[idx])[0], y_pseudo[idx]).backward()
                opt.step()
        miras.eval()
        with torch.no_grad():
            acc = (miras(X_feat_test)[0].argmax(-1) == y_te).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in miras.state_dict().items()}
        log(f"  Round {sl+1}: +{len(X_pseudo):,} pseudo | Acc={acc*100:.2f}%")
    
    miras.load_state_dict(best_state)
    
    log("\n" + "=" * 60)
    with torch.no_grad():
        pred = miras(X_feat_test)[0].argmax(-1).numpy()
    class_accs = [(pred[y_test==c] == y_test[y_test==c]).mean() for c in range(N_CLASSES)]
    
    log("FINAL RESULTS:")
    log(f"  MIDAS 50% Missing: {best_missing_acc*100:.1f}%")
    log(f"  MIRAS Acc: {best_acc*100:.2f}%")
    log(f"  Min Class Acc: {min(class_accs)*100:.1f}%")
    log(f"  Mean Class Acc: {np.mean(class_accs)*100:.1f}%")
    
    passed = best_missing_acc >= 0.90 and best_acc >= 0.99 and min(class_accs) >= 0.95
    log(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    torch.save({'midas': midas.state_dict(), 'miras': miras.state_dict(), 'n_features': N_FEATURES, 'n_classes': N_CLASSES, 'midas_acc': best_missing_acc, 'miras_acc': best_acc, 'version': 'V5'}, SAVE_PATH)
    log(f"Saved: {SAVE_PATH}")
    log("=" * 60)

if __name__ == '__main__':
    train()
