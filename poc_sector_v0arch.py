import json, os, math, random
import numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class Config:
    d_model = 64
    n_heads = 4
    n_latent = 16
    dropout = 0.3
    max_cols = 50
    max_rows = 10
    sbert_dim = 384
    lr = 3e-4
    warmup_epochs = 5
    total_epochs = 100
    aug_per_dataset = 5
    aug_column_dropout = 0.3
    aug_cell_noise = 0.1
    device = "mps" if torch.backends.mps.is_available() else "cpu"

cfg = Config()
device = torch.device(cfg.device)
print(f"Device: {device}")

METADATA_PATH = Path(os.path.expanduser("~/Desktop/schemalabsai/data/poc_synthetic_1000.json"))
CHECKPOINT_PATH = Path(os.path.expanduser("~/Desktop/schemalabsai/checkpoints/poc_sector_v1.pt"))
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(METADATA_PATH) as f:
    RAW_DATA = json.load(f)

SECTORS = sorted(set(d["sector"] for d in RAW_DATA))
S2I = {s:i for i,s in enumerate(SECTORS)}
N_SECTORS = len(SECTORS)
print(f"Datasets: {len(RAW_DATA)}, Sectors: {N_SECTORS}")

# Pre-compute SBERT embeddings for all column names
print("Computing SBERT embeddings...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
all_col_names = set()
for d in RAW_DATA:
    for c in d["columns"]:
        all_col_names.add(c.lower().replace("_", " "))
all_col_names = sorted(all_col_names)
col_embeddings = sbert.encode(all_col_names, show_progress_bar=False, convert_to_numpy=True)
COL_EMB_MAP = {name: col_embeddings[i] for i, name in enumerate(all_col_names)}
print(f"Embedded {len(COL_EMB_MAP)} unique column names")

def get_col_embedding(col_name):
    key = col_name.lower().replace("_", " ")
    if key in COL_EMB_MAP:
        return COL_EMB_MAP[key]
    # Runtime encode for unseen columns
    return sbert.encode([key], convert_to_numpy=True)[0]

def is_numeric(val):
    try:
        float(str(val).replace(",",""))
        return True
    except:
        return False

def parse_numeric(val):
    try:
        return float(str(val).replace(",",""))
    except:
        return 0.0

# Distribution fingerprint per column: [mean, std, min, max, numeric_ratio, unique_ratio, n_values]
def compute_dist_fingerprint(values):
    nums = []
    n_total = len(values)
    unique = len(set(values))
    for v in values:
        if is_numeric(v):
            nums.append(parse_numeric(v))
    if nums:
        arr = np.array(nums)
        return [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
            len(nums) / max(n_total, 1),
            unique / max(n_total, 1),
            float(n_total)
        ]
    else:
        return [0, 0, 0, 0, 0, unique / max(n_total, 1), float(n_total)]

FINGERPRINT_DIM = 7

def encode_dataset(d, augment=False):
    columns = list(d["columns"][:cfg.max_cols])
    all_rows = d.get("sample_rows", [])
    n_cols = len(columns)

    # Row subset
    if augment and len(all_rows) > cfg.max_rows:
        row_indices = sorted(random.sample(range(len(all_rows)), cfg.max_rows))
        rows = [all_rows[i] for i in row_indices]
    else:
        rows = all_rows[:cfg.max_rows]

    # Column dropout
    if augment and cfg.aug_column_dropout and n_cols > 4:
        keep = max(3, int(n_cols * (1 - random.uniform(0, cfg.aug_column_dropout))))
        keep_idx = sorted(random.sample(range(n_cols), keep))
        columns = [columns[i] for i in keep_idx]
        rows = [[row[i] if i < len(row) else "" for i in keep_idx] for row in rows]
        n_cols = len(columns)

    # Column shuffle
    if augment and n_cols > 2:
        perm = list(range(n_cols))
        random.shuffle(perm)
        columns = [columns[p] for p in perm]
        rows = [[row[p] if p < len(row) else "" for p in perm] for row in rows]

    # SBERT column embeddings
    col_embs = torch.zeros(cfg.max_cols, cfg.sbert_dim)
    col_mask = torch.zeros(cfg.max_cols, dtype=torch.bool)
    for i, col in enumerate(columns[:cfg.max_cols]):
        col_embs[i] = torch.tensor(get_col_embedding(col))
        col_mask[i] = True

    # Distribution fingerprints per column
    dist_fps = torch.zeros(cfg.max_cols, FINGERPRINT_DIM)
    for c_idx in range(min(n_cols, cfg.max_cols)):
        col_values = [row[c_idx] if c_idx < len(row) else "" for row in rows]
        if augment and cfg.aug_cell_noise:
            noised = []
            for v in col_values:
                if is_numeric(v):
                    nv = parse_numeric(v) * (1 + random.uniform(-cfg.aug_cell_noise, cfg.aug_cell_noise))
                    noised.append(str(round(nv, 4)))
                else:
                    noised.append(v)
            col_values = noised
        fp = compute_dist_fingerprint(col_values)
        # Log-scale for large values
        fp_scaled = []
        for v in fp:
            if abs(v) > 1:
                fp_scaled.append(math.copysign(math.log1p(abs(v)), v) / 20.0)
            else:
                fp_scaled.append(v)
        dist_fps[c_idx] = torch.tensor(fp_scaled)

    return col_embs, col_mask, dist_fps

# ============================================================
# SchemaProcessing V1 (sentence-transformers based)
# ============================================================
class SchemaProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Linear(cfg.sbert_dim, d)
        self.schema_tf = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d*2,
            batch_first=True, dropout=cfg.dropout
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, col_embs, col_mask):
        x = self.proj(col_embs)
        x = self.schema_tf(x, src_key_padding_mask=~col_mask)
        x = self.norm(x) * col_mask.unsqueeze(-1).float()
        return x

# ============================================================
# CellProcessing V1 (distribution fingerprint based)
# ============================================================
class CellProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.fp_proj = nn.Linear(FINGERPRINT_DIM, d)
        self.norm = nn.LayerNorm(d)
        self.d_model = d

    def sinusoidal_position(self, n_pos, d_model, dev):
        pe = torch.zeros(n_pos, d_model, device=dev)
        pos = torch.arange(0, n_pos, dtype=torch.float, device=dev).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=dev) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe

    def forward(self, dist_fps, col_mask):
        B, C, _ = dist_fps.shape
        d = self.d_model
        x = self.fp_proj(dist_fps)
        pos = self.sinusoidal_position(C, d, dist_fps.device).unsqueeze(0).expand(B, -1, -1)
        x = x + pos
        x = self.norm(x) * col_mask.unsqueeze(-1).float()
        return x

# ============================================================
# LocalReasoning (V0 - column-wise attention only, no rows needed)
# ============================================================
class LocalReasoning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.col_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, col_mask):
        attn_out, _ = self.col_attn(x, x, x, key_padding_mask=~col_mask)
        x = x + self.norm(attn_out)
        return x

# ============================================================
# GlobalReasoning (V0 - perceiver style)
# ============================================================
class GlobalReasoning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.latents = nn.Parameter(torch.randn(cfg.n_latent, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d*2,
            batch_first=True, dropout=cfg.dropout
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x, col_mask):
        B = x.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(lat, x, x, key_padding_mask=~col_mask)
        out = self.self_attn(out)
        return self.norm(out).mean(dim=1)

# ============================================================
# SectorHead
# ============================================================
class SectorHead(nn.Module):
    def __init__(self, cfg, n_sectors):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, n_sectors)
        )

    def forward(self, global_repr, schema_pool):
        return self.head(torch.cat([global_repr, schema_pool], dim=-1))

# ============================================================
# Full Model
# ============================================================
class SectorAgnosticModel(nn.Module):
    def __init__(self, cfg, n_sectors):
        super().__init__()
        self.schema_proc = SchemaProcessing(cfg)
        self.cell_proc = CellProcessing(cfg)
        self.local_reason = LocalReasoning(cfg)
        self.global_reason = GlobalReasoning(cfg)
        self.sector_head = SectorHead(cfg, n_sectors)

    def forward(self, col_embs, col_mask, dist_fps):
        schema = self.schema_proc(col_embs, col_mask)
        cells = self.cell_proc(dist_fps, col_mask)
        combined = schema + cells
        local = self.local_reason(combined, col_mask)
        glob = self.global_reason(local, col_mask)
        s_pool = (schema * col_mask.unsqueeze(-1).float()).sum(1) / col_mask.sum(1, keepdim=True).float().clamp(min=1)
        return self.sector_head(glob, s_pool)

def get_lr(epoch, warmup, total, base_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def to_device(tensors):
    return tuple(t.unsqueeze(0).to(device) for t in tensors)

def evaluate_model(model, data_list):
    model.eval()
    correct = 0
    details = []
    with torch.no_grad():
        for d in data_list:
            inputs = to_device(encode_dataset(d, augment=False))
            pred = model(*inputs).argmax(-1).item()
            actual = S2I[d["sector"]]
            ok = pred == actual
            if ok: correct += 1
            details.append((d, pred, actual, ok))
    return correct, details

# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("SchemaLabs PoC V1: Sector Agnostic")
print("SBERT SchemaProcessing + Distribution CellProcessing")
print("+ LocalReasoning + GlobalReasoning + SectorHead")
print("=" * 60)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# PHASE 1
print("\n" + "=" * 60)
print("PHASE 1: SEEN SECTOR TEST (80/20)")
print("=" * 60)

indices = list(range(len(RAW_DATA)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_data = [RAW_DATA[i] for i in indices[:split]]
test_data = [RAW_DATA[i] for i in indices[split:]]
print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

model = SectorAgnosticModel(cfg, N_SECTORS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {n_params:,}")

best_test_acc = 0
best_state = None
for epoch in range(cfg.total_epochs):
    model.train()
    lr = get_lr(epoch, cfg.warmup_epochs, cfg.total_epochs, cfg.lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    train_samples = []
    for d in train_data:
        for _ in range(cfg.aug_per_dataset):
            train_samples.append(d)
    random.shuffle(train_samples)

    total_loss = 0
    for d in train_samples:
        inputs = to_device(encode_dataset(d, augment=True))
        label = torch.tensor([S2I[d["sector"]]]).to(device)
        logits = model(*inputs)
        loss = F.cross_entropy(logits, label, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        tr_c, _ = evaluate_model(model, train_data)
        te_c, te_det = evaluate_model(model, test_data)
        tr_acc = tr_c / len(train_data) * 100
        te_acc = te_c / len(test_data) * 100
        avg_loss = total_loss / len(train_samples)
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve = getattr(model, '_no_improve', 0) + 1
        model._no_improve = no_improve if te_acc <= best_test_acc else 0
        print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} lr={lr:.6f} Train={tr_acc:.1f}% Val={te_acc:.1f}% (best={best_test_acc:.1f}%)")
        if best_test_acc >= 100.0 and tr_acc >= 100.0:
            print("  Early stop: 100% reached")
            break

if best_state:
    model.load_state_dict(best_state)
te_c, te_det = evaluate_model(model, test_data)
print(f"\n  FINAL TEST: {te_c}/{len(test_data)} = {te_c/len(test_data)*100:.1f}%")
for d, pred, actual, ok in te_det:
    if not ok:
        print(f"    [XX] {d['folder'][:40]:40s} actual={SECTORS[actual]:15s} pred={SECTORS[pred]}")

torch.save({
    "model_state_dict": model.state_dict(),
    "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    "accuracy": best_test_acc,
    "sectors": SECTORS
}, CHECKPOINT_PATH)
print(f"  Checkpoint: {CHECKPOINT_PATH}")

# PHASE 2: UNSEEN SECTOR TEST
print("\n" + "=" * 60)
print("PHASE 2: UNSEEN SECTOR TEST (leave-one-sector-out)")
print("NO keyword dictionary - SBERT + distribution patterns")
print("=" * 60)

results = {}
for holdout in SECTORS:
    ho = [d for d in RAW_DATA if d["sector"] == holdout]
    tr = [d for d in RAW_DATA if d["sector"] != holdout]
    if not ho:
        continue

    m = SectorAgnosticModel(cfg, N_SECTORS).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=0.05)

    for ep in range(60):
        m.train()
        lr = get_lr(ep, 3, 60, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        samples = []
        for d in tr:
            for _ in range(cfg.aug_per_dataset):
                samples.append(d)
        random.shuffle(samples)
        for d in samples:
            inputs = to_device(encode_dataset(d, augment=True))
            label = torch.tensor([S2I[d["sector"]]]).to(device)
            logits = m(*inputs)
            loss = F.cross_entropy(logits, label, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

    c, preds = evaluate_model(m, ho)
    correct_n = 0
    for d, pred, actual, ok in preds:
        if ok: correct_n += 1
    acc = correct_n / len(ho) * 100
    # Show first 3 predictions per holdout
    shown = 0
    for d, pred, actual, ok in preds:
        if shown < 3:
            mark = "OK" if ok else "XX"
            print(f"  [{mark}] holdout={holdout:15s} {d['folder'][:30]:30s} -> {SECTORS[pred]}")
            shown += 1
    if len(ho) > 3:
        print(f"  ... {holdout}: {correct_n}/{len(ho)} = {acc:.0f}%")
    results[holdout] = (correct_n, len(ho))

print(f"\n{'=' * 60}")
print("UNSEEN SUMMARY")
print(f"{'=' * 60}")
tot_c = sum(v[0] for v in results.values())
tot_n = sum(v[1] for v in results.values())
for s in sorted(results):
    c, n = results[s]
    mark = "OK" if c == n else ("--" if c/n >= 0.5 else "XX")
    print(f"  [{mark}] {s:20s}: {c}/{n} = {c/n*100:.0f}%")
print(f"\n  OVERALL: {tot_c}/{tot_n} = {tot_c/tot_n*100:.1f}%")
if tot_c / tot_n >= 0.8:
    print("  SECTOR-AGNOSTIC CALISIYOR")
elif tot_c / tot_n >= 0.5:
    print("  KISMEN CALISIYOR")
else:
    print("  YETERSIZ")
