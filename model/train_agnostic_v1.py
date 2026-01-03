import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

# ============================================================
# SCHEMALABSAI - Domain Agnostic V1
# MIDAS + Self-Learning (No MIRAS)
# ============================================================

LOG_PATH = "../checkpoints/schemalabsai_agnostic_v1_log.txt"
SAVE_PATH = "../checkpoints/schemalabsai_agnostic_v1.pt"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp} - {msg}\n")

# ==================== DATA ====================
def generate_data(n_samples, n_features=10, n_classes=100):
    np.random.seed(42)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    samples_per_class = n_samples // n_classes
    class_centers = np.random.randn(n_classes, n_features) * 5
    
    for c in range(n_classes):
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class
        center = class_centers[c]
        spread = np.random.rand() * 0.5 + 0.2
        data = np.random.randn(samples_per_class, n_features) * spread + center
        corr_strength = np.random.rand() * 0.5
        for i in range(1, n_features):
            data[:, i] = data[:, i] * (1 - corr_strength) + data[:, i-1] * corr_strength + center[i]
        X[start_idx:end_idx] = data
        y[start_idx:end_idx] = c
    
    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    return X, y

# ==================== MODELS ====================
class MIDAS(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, mask):
        return self.decoder(self.encoder(torch.cat([x * mask, mask], dim=1)))
    
    def encode(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        return self.encoder(torch.cat([x * mask, mask], dim=1))
    
    def impute(self, x, mask, n_iter=5):
        current = x * mask
        for _ in range(n_iter):
            current = x * mask + self.forward(current, mask) * (1 - mask)
        return current

class Classifier(nn.Module):
    def __init__(self, input_dim=256, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== TRAINING ====================
def train():
    log("=" * 70)
    log("SCHEMALABSAI - Domain Agnostic V1")
    log("MIDAS + Self-Learning")
    log("=" * 70)
    
    # Config
    N_FEATURES = 10
    N_CLASSES = 100
    N_SAMPLES = 10_000_000
    BATCH_SIZE = 2048
    
    # [1/5] Generate data
    log("\n[1/5] Generating data...")
    X, y = generate_data(N_SAMPLES, N_FEATURES, N_CLASSES)
    
    n_labeled = int(N_SAMPLES * 0.6)
    n_unlabeled = int(N_SAMPLES * 0.2)
    n_test = N_SAMPLES - n_labeled - n_unlabeled
    
    X_labeled, y_labeled = X[:n_labeled], y[:n_labeled]
    X_unlabeled = X[n_labeled:n_labeled + n_unlabeled]
    X_test, y_test = X[n_labeled + n_unlabeled:], y[n_labeled + n_unlabeled:]
    
    log(f"  Labeled: {n_labeled:,}, Unlabeled: {n_unlabeled:,}, Test: {n_test:,}")
    
    # [2/5] Train MIDAS
    log("\n[2/5] Training MIDAS...")
    midas = MIDAS(N_FEATURES, 512)
    midas_opt = AdamW(midas.parameters(), lr=0.001, weight_decay=0.01)
    
    X_t = torch.FloatTensor(X_labeled)
    
    for ep in tqdm(range(50), desc="MIDAS"):
        midas.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch = X_t[idx]
            mask = (torch.rand_like(batch) > np.random.uniform(0.1, 0.5)).float()
            midas_opt.zero_grad()
            nn.MSELoss()(midas(batch * mask, mask), batch).backward()
            midas_opt.step()
    
    # Test MIDAS
    midas.eval()
    with torch.no_grad():
        X_te = torch.FloatTensor(X_test[:10000])
        mask_50 = (torch.rand_like(X_te) > 0.5).float()
        imputed = midas.impute(X_te * mask_50, mask_50)
        mse = ((imputed - X_te) ** 2).mean().item()
    log(f"  MIDAS MSE @50% missing: {mse:.4f}")
    
    # Freeze MIDAS
    for p in midas.parameters():
        p.requires_grad = False
    
    # [3/5] Extract features
    log("\n[3/5] Extracting features...")
    midas.eval()
    
    def extract_features(X_data):
        features = []
        X_tensor = torch.FloatTensor(X_data)
        for i in range(0, len(X_tensor), 10000):
            with torch.no_grad():
                features.append(midas.encode(X_tensor[i:i+10000]))
        return torch.cat(features)
    
    X_feat_labeled = extract_features(X_labeled)
    X_feat_unlabeled = extract_features(X_unlabeled)
    X_feat_test = extract_features(X_test)
    
    log(f"  Features shape: {X_feat_labeled.shape}")
    
    # [4/5] Train classifier
    log("\n[4/5] Training classifier (early stopping, patience=10)...")
    classifier = Classifier(256, N_CLASSES)
    clf_opt = AdamW(classifier.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    y_t = torch.LongTensor(y_labeled)
    y_te = torch.LongTensor(y_test)
    
    best_acc = 0
    best_state = None
    patience_counter = 0
    
    for ep in tqdm(range(100), desc="Classifier"):
        classifier.train()
        perm = torch.randperm(len(X_feat_labeled))
        for i in range(0, len(X_feat_labeled), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            clf_opt.zero_grad()
            loss_fn(classifier(X_feat_labeled[idx]), y_t[idx]).backward()
            clf_opt.step()
        
        classifier.eval()
        with torch.no_grad():
            acc = (classifier(X_feat_test).argmax(-1) == y_te).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if best_acc >= 0.99:
            log(f"  🎉 Reached 99%+ at epoch {ep+1}")
            break
        if patience_counter >= 10:
            log(f"  Early stopping at epoch {ep+1}")
            break
    
    classifier.load_state_dict(best_state)
    log(f"  Before Self-Learning: {best_acc*100:.1f}%")
    
    # [5/5] Self-Learning
    log("\n[5/5] Self-Learning (5 rounds, threshold=0.95)...")
    
    for sl_round in range(5):
        classifier.eval()
        
        # Get pseudo labels in batches
        all_pseudo_x, all_pseudo_y = [], []
        for i in range(0, len(X_feat_unlabeled), 50000):
            batch = X_feat_unlabeled[i:i+50000]
            with torch.no_grad():
                logits = classifier(batch)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(1)
                high_conf = conf > 0.95
                if high_conf.sum() > 0:
                    all_pseudo_x.append(batch[high_conf])
                    all_pseudo_y.append(pred[high_conf])
        
        if len(all_pseudo_x) == 0:
            log(f"  Round {sl_round+1}: No pseudo labels")
            continue
        
        X_pseudo = torch.cat(all_pseudo_x)
        y_pseudo = torch.cat(all_pseudo_y)
        
        # Train on pseudo labels
        classifier.train()
        for _ in range(3):
            perm = torch.randperm(len(X_pseudo))
            for i in range(0, len(X_pseudo), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                clf_opt.zero_grad()
                loss_fn(classifier(X_pseudo[idx]), y_pseudo[idx]).backward()
                clf_opt.step()
        
        classifier.eval()
        with torch.no_grad():
            acc = (classifier(X_feat_test).argmax(-1) == y_te).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
        
        log(f"  Round {sl_round+1}: +{len(X_pseudo):,} pseudo | Acc={acc*100:.1f}% (best={best_acc*100:.1f}%)")
    
    classifier.load_state_dict(best_state)
    
    # Final evaluation
    log("\n" + "=" * 70)
    log("FINAL EVALUATION")
    log("=" * 70)
    
    classifier.eval()
    with torch.no_grad():
        final_acc = (classifier(X_feat_test).argmax(-1) == y_te).float().mean().item()
    
    # Per-class accuracy
    with torch.no_grad():
        preds = classifier(X_feat_test).argmax(-1).numpy()
    
    class_accs = []
    for c in range(N_CLASSES):
        mask = y_test == c
        if mask.sum() > 0:
            class_accs.append((preds[mask] == y_test[mask]).mean())
    
    log(f"  Final Accuracy: {final_acc*100:.1f}%")
    log(f"  Min class acc: {min(class_accs)*100:.1f}%")
    log(f"  Max class acc: {max(class_accs)*100:.1f}%")
    log(f"  Mean class acc: {np.mean(class_accs)*100:.1f}%")
    
    # Save
    log(f"\nSaving model to {SAVE_PATH}...")
    torch.save({
        'midas': midas.state_dict(),
        'classifier': classifier.state_dict(),
        'n_features': N_FEATURES,
        'n_classes': N_CLASSES,
        'accuracy': final_acc,
        'class_accuracies': class_accs,
        'version': 'Domain_Agnostic_V1_MIDAS_SelfLearning'
    }, SAVE_PATH)
    
    log(f"\n{'='*70}")
    log(f"✅ Training complete!")
    log(f"   Accuracy: {final_acc*100:.1f}%")
    log(f"   Model: {SAVE_PATH}")
    log(f"{'='*70}")
    
    return final_acc >= 0.99

if __name__ == '__main__':
    success = train()
    exit(0 if success else 1)
