#!/usr/bin/env python3
"""
SchemaLabs V1 — FAST Production Training (All-in-One)
Auto pre-computes encodings if not found, then trains with full GPU power.

Optimizations:
  - Pre-computed SBERT encodings (one-time, saved to disk)
  - DataLoader: batch=32, num_workers=4, pin_memory, prefetch
  - AMP (FP16 mixed precision) — T4 2x throughput
  - GPU-side augmentation
  - cudnn.benchmark, TF32
  - non_blocking transfers

Target: 6/s → 150-300/s, 1 epoch: 22h → 25-50min

Usage:
  python schema_v1_production.py              # auto pre-compute + train
  python schema_v1_production.py --precompute # force re-precompute
"""
import json, os, math, random, time, gc, sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AMP setup — compatible with PyTorch 1.x and 2.x
if torch.cuda.is_available():
    from torch.cuda.amp import autocast as _autocast, GradScaler
    def amp_autocast():
        return _autocast()
else:
    from contextlib import nullcontext
    GradScaler = None
    def amp_autocast():
        return nullcontext()

from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
class Config:
    d_model = 640
    n_heads = 16
    n_latent = 128
    n_layers = 6
    dropout = 0.1
    sbert_dim = 384
    fingerprint_dim = 7
    max_cols = 30
    max_rows = 10

    # Training
    batch_size = 1 if not torch.cuda.is_available() else 32
    lr = 1e-4
    warmup_epochs = 3
    total_epochs = 20
    label_smoothing = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    grad_accum = 8 if not torch.cuda.is_available() else 1  # effective batch=8 on MPS

    # Augmentation
    aug_cell_noise = 0.1
    aug_column_dropout = 0.3

    # MIDAS
    midas_iterations = 10
    midas_weight = 0.1

    # MCM
    mcm_mask_ratio = 0.15
    mcm_weight = 0.1

    # MIRAS
    miras_weight = 0.05
    miras_low_rank_k = 64

    # EWC
    ewc_lambda = 1000

    # Backbone freeze
    backbone_freeze_after = 250000

    # Logging — adjusted per device
    log_every = 500 if not torch.cuda.is_available() else 1000
    acc_every = 2500 if not torch.cuda.is_available() else 5000
    checkpoint_every = 5000 if not torch.cuda.is_available() else 3000
    val_samples = 500 if not torch.cuda.is_available() else 1000

    # Paths
    sector_list_path = "data/sector_list_10000.json"
    sector_emb_path = "data/sector_embeddings_10000.npy"
    data_path = "data/v1_training_data.json"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

cfg = Config()
device = torch.device(cfg.device)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

BASE = Path(os.path.expanduser("~/Desktop/schemalabsai")) if Path(os.path.expanduser("~/Desktop/schemalabsai")).exists() else Path("/opt/schemalabsai")
PRECOMP_DIR = BASE / "data" / "v1_precomputed"
CHECKPOINT_PATH = BASE / "checkpoints" / "schema_v1_production.pt"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# PHASE 1: PRE-COMPUTE (if needed)
# ============================================================
def needs_precompute():
    if "--precompute" in sys.argv:
        return True
    required = ["col_embs.pt", "col_mask.pt", "dist_fps.pt",
                 "cell_values.pt", "cell_mask.pt", "cell_is_numeric.pt",
                 "labels.pt", "sector_emb_matrix.pt", "metadata.json"]
    return not all((PRECOMP_DIR / f).exists() for f in required)

def run_precompute():
    PRECOMP_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 1: PRE-COMPUTING ENCODINGS")
    print("=" * 60)

    # Load data
    print("Loading data...")
    t0 = time.time()
    with open(BASE / cfg.data_path) as f:
        ALL_DATA = json.load(f)
    print(f"Loaded {len(ALL_DATA):,} datasets in {time.time()-t0:.1f}s")

    with open(BASE / cfg.sector_list_path) as f:
        sector_data = json.load(f)
        ALL_SECTORS = sector_data["sectors"]
        HIERARCHY = sector_data["hierarchy"]

    MAIN_SECTORS = sorted(HIERARCHY.keys())
    MAIN_S2I = {s: i for i, s in enumerate(MAIN_SECTORS)}

    for d in ALL_DATA:
        sector = d.get("main_sector", d.get("sector", "manufacturing"))
        if sector not in MAIN_S2I:
            for main in MAIN_SECTORS:
                if sector in main or main in sector:
                    sector = main
                    break
            else:
                sector = "manufacturing"
        d["main_sector"] = sector

    DS_SECTORS = sorted(set(d["main_sector"] for d in ALL_DATA))
    DS_S2I = {s: i for i, s in enumerate(DS_SECTORS)}
    print(f"Dataset sectors: {len(DS_SECTORS)}")

    # SBERT
    print("Loading SBERT...")
    dev_sbert = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=dev_sbert)

    print("Collecting unique column names...")
    all_col_names = set()
    for d in ALL_DATA:
        for c in d["columns"]:
            all_col_names.add(c.lower().replace("_", " "))
    all_col_names = sorted(all_col_names)
    print(f"Unique columns: {len(all_col_names):,}")

    print("Encoding columns...")
    COL_EMB_MAP = {}
    BATCH = 1024
    for i in range(0, len(all_col_names), BATCH):
        batch = all_col_names[i:i+BATCH]
        embs = sbert.encode(batch, show_progress_bar=False, convert_to_numpy=True, batch_size=1024)
        for name, emb in zip(batch, embs):
            COL_EMB_MAP[name] = emb
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i+BATCH, len(all_col_names)):,}/{len(all_col_names):,}")
    print(f"Embedded {len(COL_EMB_MAP):,} columns")

    # Sector embeddings
    SECTOR_EMBS = np.load(BASE / cfg.sector_emb_path)
    SUB_TO_MAIN = {}
    for main, subs in HIERARCHY.items():
        SUB_TO_MAIN[main] = main
        for s in subs:
            SUB_TO_MAIN[s] = main

    MAIN_SECTOR_EMBS = {}
    for main in MAIN_SECTORS:
        indices = [i for i, s in enumerate(ALL_SECTORS) if SUB_TO_MAIN.get(s) == main]
        if indices:
            MAIN_SECTOR_EMBS[main] = np.mean(SECTOR_EMBS[indices], axis=0)
        else:
            MAIN_SECTOR_EMBS[main] = sbert.encode([main], convert_to_numpy=True)[0]

    DS_SECTOR_MATRIX = np.array([MAIN_SECTOR_EMBS.get(s, sbert.encode([s], convert_to_numpy=True)[0]) for s in DS_SECTORS])
    torch.save(torch.tensor(DS_SECTOR_MATRIX, dtype=torch.float32), PRECOMP_DIR / "sector_emb_matrix.pt")

    del sbert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Encode helpers
    def is_numeric(val):
        try:
            float(str(val).replace(",", ""))
            return True
        except:
            return False

    def parse_numeric(val):
        try:
            return float(str(val).replace(",", ""))
        except:
            return 0.0

    def compute_fingerprint(values):
        nums = [parse_numeric(v) for v in values if is_numeric(v)]
        n_total = len(values)
        unique = len(set(values))
        if nums:
            arr = np.array(nums)
            fp = [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)),
                  float(np.max(arr)), len(nums)/max(n_total,1),
                  unique/max(n_total,1), float(n_total)]
        else:
            fp = [0, 0, 0, 0, 0, unique/max(n_total,1), float(n_total)]
        return [math.copysign(math.log1p(abs(v)), v)/20.0 if abs(v) > 1 else v for v in fp]

    # Pre-compute all
    N = len(ALL_DATA)
    print(f"\nEncoding {N:,} samples...")

    all_col_embs = torch.zeros(N, cfg.max_cols, cfg.sbert_dim, dtype=torch.float16)
    all_col_mask = torch.zeros(N, cfg.max_cols, dtype=torch.bool)
    all_dist_fps = torch.zeros(N, cfg.max_cols, cfg.fingerprint_dim, dtype=torch.float16)
    all_cell_values = torch.zeros(N, cfg.max_rows, cfg.max_cols, dtype=torch.float16)
    all_cell_mask = torch.zeros(N, cfg.max_rows, cfg.max_cols, dtype=torch.bool)
    all_cell_is_numeric = torch.zeros(N, cfg.max_rows, cfg.max_cols, dtype=torch.bool)
    all_labels = torch.zeros(N, dtype=torch.long)

    t_start = time.time()
    for i, d in enumerate(ALL_DATA):
        columns = list(d["columns"][:cfg.max_cols])
        rows = d.get("sample_rows", [])[:cfg.max_rows]
        n_cols = len(columns)

        for ci, col in enumerate(columns[:cfg.max_cols]):
            key = col.lower().replace("_", " ")
            if key in COL_EMB_MAP:
                all_col_embs[i, ci] = torch.tensor(COL_EMB_MAP[key], dtype=torch.float16)
            all_col_mask[i, ci] = True

        for c_idx in range(min(n_cols, cfg.max_cols)):
            col_vals = [row[c_idx] if c_idx < len(row) else "" for row in rows]
            fp = compute_fingerprint(col_vals)
            all_dist_fps[i, c_idx] = torch.tensor(fp, dtype=torch.float16)
            for r_idx, v in enumerate(col_vals[:cfg.max_rows]):
                if v and str(v).strip():
                    all_cell_mask[i, r_idx, c_idx] = True
                    if is_numeric(v):
                        val = parse_numeric(v)
                        all_cell_values[i, r_idx, c_idx] = math.copysign(math.log1p(abs(val)), val) / 20.0
                        all_cell_is_numeric[i, r_idx, c_idx] = True

        all_labels[i] = DS_S2I[d["main_sector"]]

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  [{i+1:,}/{N:,}] rate={rate:.0f}/s eta={eta:.0f}s")
        if (i + 1) % 100000 == 0:
            gc.collect()

    print(f"Encoding done in {time.time()-t_start:.0f}s")

    # Save
    print("Saving tensors...")
    torch.save(all_col_embs, PRECOMP_DIR / "col_embs.pt")
    torch.save(all_col_mask, PRECOMP_DIR / "col_mask.pt")
    torch.save(all_dist_fps, PRECOMP_DIR / "dist_fps.pt")
    torch.save(all_cell_values, PRECOMP_DIR / "cell_values.pt")
    torch.save(all_cell_mask, PRECOMP_DIR / "cell_mask.pt")
    torch.save(all_cell_is_numeric, PRECOMP_DIR / "cell_is_numeric.pt")
    torch.save(all_labels, PRECOMP_DIR / "labels.pt")

    meta = {
        "n_samples": N,
        "ds_sectors": DS_SECTORS,
        "ds_s2i": DS_S2I,
        "max_cols": cfg.max_cols,
        "max_rows": cfg.max_rows,
        "sbert_dim": cfg.sbert_dim,
        "fp_dim": cfg.fingerprint_dim,
    }
    with open(PRECOMP_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = sum((PRECOMP_DIR / fn).stat().st_size for fn in os.listdir(PRECOMP_DIR) if fn.endswith(".pt"))
    print(f"Saved to {PRECOMP_DIR} ({total_bytes / 1024**2:.0f} MB)")
    print("Pre-compute DONE.\n")

    del all_col_embs, all_col_mask, all_dist_fps, all_cell_values, all_cell_mask, all_cell_is_numeric, all_labels
    del COL_EMB_MAP, ALL_DATA
    gc.collect()

# Run pre-compute if needed
if needs_precompute():
    run_precompute()
else:
    print("Pre-computed data found, skipping to training.")

# ============================================================
# PHASE 2: LOAD PRE-COMPUTED DATA
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: LOADING PRE-COMPUTED DATA")
print("=" * 60)

t0 = time.time()
col_embs_all = torch.load(PRECOMP_DIR / "col_embs.pt", weights_only=True)
col_mask_all = torch.load(PRECOMP_DIR / "col_mask.pt", weights_only=True)
dist_fps_all = torch.load(PRECOMP_DIR / "dist_fps.pt", weights_only=True)
cell_values_all = torch.load(PRECOMP_DIR / "cell_values.pt", weights_only=True)
cell_mask_all = torch.load(PRECOMP_DIR / "cell_mask.pt", weights_only=True)
cell_is_num_all = torch.load(PRECOMP_DIR / "cell_is_numeric.pt", weights_only=True)
labels_all = torch.load(PRECOMP_DIR / "labels.pt", weights_only=True)
sector_emb_matrix = torch.load(PRECOMP_DIR / "sector_emb_matrix.pt", weights_only=True).to(device)

with open(PRECOMP_DIR / "metadata.json") as f:
    meta = json.load(f)
DS_SECTORS = meta["ds_sectors"]
DS_S2I = meta["ds_s2i"]
N_DS = len(DS_SECTORS)
N = len(labels_all)
print(f"Loaded {N:,} samples in {time.time()-t0:.1f}s — {N_DS} sectors")

# ============================================================
# DATASET + DATALOADER
# ============================================================
class PrecomputedDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        return (col_embs_all[i].float(), col_mask_all[i], dist_fps_all[i].float(),
                cell_values_all[i].float(), cell_mask_all[i].float(),
                cell_is_num_all[i].float(), labels_all[i])

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

all_indices = list(range(N))
random.shuffle(all_indices)
split = int(0.95 * N)
train_indices = all_indices[:split]
test_indices = all_indices[split:]

_is_cuda = torch.cuda.is_available()

train_loader = DataLoader(
    PrecomputedDataset(train_indices), batch_size=cfg.batch_size, shuffle=True,
    num_workers=4 if _is_cuda else 0, pin_memory=_is_cuda,
    persistent_workers=True if _is_cuda else False,
    prefetch_factor=4 if _is_cuda else None, drop_last=True)

test_loader = DataLoader(
    PrecomputedDataset(test_indices), batch_size=cfg.batch_size if not _is_cuda else cfg.batch_size * 2, shuffle=False,
    num_workers=2 if _is_cuda else 0, pin_memory=_is_cuda,
    persistent_workers=True if _is_cuda else False)

print(f"Train: {len(train_indices):,}, Test: {len(test_indices):,}")
print(f"Batches/epoch: {len(train_loader):,}, Batch size: {cfg.batch_size}")

# ============================================================
# GPU AUGMENTATION
# ============================================================
def gpu_augment(ce, cm, df, cv, cmask, cin):
    B, C_max = cm.shape
    cm_bool = cm.bool()

    if cfg.aug_column_dropout > 0:
        drop = (torch.rand(B, C_max, device=ce.device) < cfg.aug_column_dropout) & cm_bool
        for b in range(B):
            active = cm_bool[b].sum().item()
            if active <= 3:
                drop[b] = False
                continue
            dropped = drop[b].sum().item()
            if active - dropped < 3:
                drop_idx = drop[b].nonzero().squeeze(-1)
                n_undrop = 3 - int(active - dropped)
                undrop = drop_idx[torch.randperm(len(drop_idx), device=ce.device)[:n_undrop]]
                drop[b, undrop] = False
        keep = ~drop
        ce = ce * keep.unsqueeze(-1).float()
        cm = cm & keep
        df = df * keep.unsqueeze(-1).float()
        cv = cv * keep.unsqueeze(1).float()
        cmask = cmask * keep.unsqueeze(1).float()
        cin = cin * keep.unsqueeze(1).float()

    if cfg.aug_cell_noise > 0:
        noise = 1.0 + (torch.rand_like(cv) * 2 - 1) * cfg.aug_cell_noise
        cv = cv * noise * cin + cv * (1 - cin)

    return ce, cm, df, cv, cmask, cin

# ============================================================
# MODEL COMPONENTS
# ============================================================
class MIDAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.imputer = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d*4, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.denoiser = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, 1))
        self.norm = nn.LayerNorm(d)
        self.iterations = cfg.midas_iterations

    def forward(self, x, cell_mask):
        mask_bool = cell_mask.unsqueeze(-1).bool().expand_as(x)
        for _ in range(self.iterations):
            imputed = self.imputer(x)
            x = torch.where(mask_bool, x, imputed)
        x = self.norm(x)
        recon = self.denoiser(x).squeeze(-1)
        return x, recon

class CellProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.value_proj = nn.Linear(1, d)
        self.numeric_embed = nn.Embedding(2, d)
        self.fp_proj = nn.Linear(cfg.fingerprint_dim, d)
        self.fusion = nn.Linear(d * 3, d)
        self.norm = nn.LayerNorm(d)
        self.d_model = d
        # Pre-compute positional encoding (static)
        pe = torch.zeros(cfg.max_cols, d)
        pos = torch.arange(0, cfg.max_cols, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d // 2])
        self.register_buffer('pe', pe)

    def forward(self, cell_values, cell_is_numeric, dist_fps, col_mask):
        B, R, C = cell_values.shape
        d = self.d_model
        val_emb = self.value_proj(cell_values.unsqueeze(-1))
        type_emb = self.numeric_embed(cell_is_numeric.long())
        fp_emb = self.fp_proj(dist_fps).unsqueeze(1).expand(B, R, C, d)
        pos = self.pe[:C].unsqueeze(0).unsqueeze(0).expand(B, R, -1, -1)
        fused = self.fusion(torch.cat([val_emb + pos, type_emb, fp_emb], dim=-1))
        fused = self.norm(fused)
        fused = fused * col_mask.unsqueeze(1).unsqueeze(-1).float()
        return fused

class SchemaProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.sbert_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model*4,
            batch_first=True, dropout=cfg.dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, col_embs, col_mask):
        x = self.proj(col_embs)
        x = self.transformer(x, src_key_padding_mask=~col_mask)
        return self.norm(x) * col_mask.unsqueeze(-1).float()

class AxialAttentionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.row_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d*4, d))
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x, col_mask):
        B, R, C, d = x.shape
        xr = x.reshape(B*R, C, d)
        mr = (~col_mask).unsqueeze(1).expand(B, R, C).reshape(B*R, C)
        a1, _ = self.row_attn(xr, xr, xr, key_padding_mask=mr)
        x = x + self.norm1(a1.view(B, R, C, d))
        xc = x.permute(0, 2, 1, 3).reshape(B*C, R, d)
        a2, _ = self.col_attn(xc, xc, xc)
        x = x + self.norm2(a2.view(B, C, R, d).permute(0, 2, 1, 3))
        x = x + self.norm3(self.ffn(x))
        return x

class LocalReasoning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([AxialAttentionLayer(cfg) for _ in range(cfg.n_layers)])
    def forward(self, x, col_mask):
        for layer in self.layers:
            x = layer(x, col_mask)
        return x

class PerceiverLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d*4, d))
        self.norm3 = nn.LayerNorm(d)

    def forward(self, latents, kv, kv_mask):
        a1, _ = self.cross_attn(latents, kv, kv, key_padding_mask=kv_mask)
        latents = latents + self.norm1(a1)
        a2, _ = self.self_attn(latents, latents, latents)
        latents = latents + self.norm2(a2)
        latents = latents + self.norm3(self.ffn(latents))
        return latents

class GlobalReasoning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.latents = nn.Parameter(torch.randn(cfg.n_latent, d) * 0.02)
        self.layers = nn.ModuleList([PerceiverLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, col_mask):
        B, R, C, d = x.shape
        flat = x.reshape(B, R*C, d)
        mask = ~col_mask.unsqueeze(1).expand(B, R, C).reshape(B, R*C)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            lat = layer(lat, flat, mask)
        return self.norm(lat).mean(dim=1)

class SectorHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(
            nn.Linear(d*2, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.sbert_dim))

    def forward(self, global_repr, schema_pool, sector_emb_matrix):
        combined = torch.cat([global_repr, schema_pool], dim=-1)
        projected = F.normalize(self.proj(combined), dim=-1)
        sector_emb = F.normalize(sector_emb_matrix, dim=-1)
        return projected @ sector_emb.t() * 10

class ClassificationHead(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, n_classes))
    def forward(self, x):
        return self.head(x)

class MCM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d*2, 1))
        self.mask_ratio = cfg.mcm_mask_ratio

    def apply_mask(self, cell_emb, cell_mask):
        B, R, C, d = cell_emb.shape
        rand = torch.rand(B, R, C, device=cell_emb.device)
        mcm_mask = (rand < self.mask_ratio) & cell_mask.unsqueeze(1).expand(B, R, C).bool()
        masked_emb = cell_emb.clone()
        masked_emb[mcm_mask] = self.mask_token
        return masked_emb, mcm_mask

    def predict(self, hidden, mcm_mask, original_values):
        pred = self.predictor(hidden).squeeze(-1)
        if mcm_mask.sum() > 0:
            return F.mse_loss(pred[mcm_mask], original_values[mcm_mask])
        return torch.tensor(0.0, device=hidden.device)

class MIRAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        k = cfg.miras_low_rank_k
        self.huber_bias = nn.Parameter(torch.zeros(d))
        self.huber_delta = 1.0
        self.retention_gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.gd_lr = nn.Parameter(torch.tensor(0.01))
        self.eta = nn.Parameter(torch.ones(d))
        self.delta_param = nn.Parameter(torch.zeros(d))
        self.alpha = nn.Parameter(torch.ones(d) * 0.5)
        self.low_rank_down = nn.Linear(d, k, bias=False)
        self.low_rank_up = nn.Linear(k, d, bias=False)
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.l2_weight = nn.Parameter(torch.tensor(0.001))
        self.rms_norm = nn.LayerNorm(d)

    def forward(self, x):
        residual = x
        diff = x - self.huber_bias
        huber = torch.where(diff.abs() <= self.huber_delta, 0.5*diff**2, self.huber_delta*(diff.abs()-0.5*self.huber_delta))
        x = x - 0.01 * huber.sign() * huber.abs().clamp(max=1.0)
        x = x * self.retention_gate(x)
        x = self.eta * x + self.delta_param
        low = self.low_rank_up(self.low_rank_down(x))
        x = self.alpha * x + (1 - self.alpha) * low
        gate_out = self.gate(torch.cat([x, residual], dim=-1))
        x = gate_out * x + (1 - gate_out) * residual
        return self.rms_norm(x)

    def get_loss(self, x):
        r = self.retention_gate(x)
        entropy = -(r * torch.log(r+1e-8) + (1-r) * torch.log(1-r+1e-8))
        return -entropy.mean() * 0.01 + self.l2_weight * (self.eta**2).mean()

# ============================================================
# EWC
# ============================================================
class EWC:
    def __init__(self, model, cfg):
        self.model = model
        self.lam = cfg.ewc_lambda
        self.params = {}
        self.fisher = {}

    def register(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.data.clone()

    def compute_fisher(self, loader, sector_emb, n_batches=50):
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for batch in loader:
            if count >= n_batches:
                break
            self.model.zero_grad()
            ce, cm, df, cv, cmask, cin, lbl = [t.to(device, non_blocking=True) for t in batch]
            sl, cl, _, _, _ = self.model(ce, cm.bool(), df, cv, cmask, cin, sector_emb, training=True)
            F.cross_entropy(sl, lbl).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data**2
            count += 1
        for n in self.fisher:
            self.fisher[n] /= count

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.params and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.lam * loss

# ============================================================
# FULL MODEL
# ============================================================
class SchemaV1Production(nn.Module):
    def __init__(self, cfg, n_sectors):
        super().__init__()
        self.cfg = cfg
        self.midas = MIDAS(cfg)
        self.cell_proc = CellProcessing(cfg)
        self.schema_proc = SchemaProcessing(cfg)
        self.local_reason = LocalReasoning(cfg)
        self.global_reason = GlobalReasoning(cfg)
        self.sector_head = SectorHead(cfg)
        self.cls_head = ClassificationHead(cfg, n_sectors)
        self.mcm = MCM(cfg)
        self.miras = MIRAS(cfg)

    def forward(self, col_embs, col_mask, dist_fps, cell_values, cell_mask, cell_is_numeric, sector_emb_matrix, training=False):
        schema = self.schema_proc(col_embs, col_mask)
        cells = self.cell_proc(cell_values, cell_is_numeric, dist_fps, col_mask)
        cells, midas_recon = self.midas(cells, cell_mask)
        cells = cells + schema.unsqueeze(1)

        mcm_mask = None
        if training:
            cells_input, mcm_mask = self.mcm.apply_mask(cells, col_mask)
        else:
            cells_input = cells

        local_out = self.local_reason(cells_input, col_mask)
        B, R, C, d = local_out.shape
        miras_in = local_out.reshape(B, R*C, d)
        local_out = self.miras(miras_in).reshape(B, R, C, d)

        global_repr = self.global_reason(local_out, col_mask)
        schema_pool = (schema * col_mask.unsqueeze(-1).float()).sum(1) / col_mask.sum(1, keepdim=True).float().clamp(min=1)

        sector_logits = self.sector_head(global_repr, schema_pool, sector_emb_matrix)
        cls_logits = self.cls_head(global_repr)

        mcm_loss = torch.tensor(0.0, device=col_embs.device)
        if training and mcm_mask is not None:
            mcm_loss = self.mcm.predict(local_out, mcm_mask, cell_values.unsqueeze(-1).expand_as(local_out)[..., 0])

        miras_loss = self.miras.get_loss(miras_in) if training else torch.tensor(0.0, device=col_embs.device)
        midas_loss = F.mse_loss(midas_recon, cell_values) if training else torch.tensor(0.0, device=col_embs.device)

        return sector_logits, cls_logits, mcm_loss, miras_loss, midas_loss

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if "sector_head" not in name and "cls_head" not in name:
                param.requires_grad = False
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Backbone frozen — {n:,} trainable params remain")

# ============================================================
# LR SCHEDULE
# ============================================================
def get_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, sector_emb, max_batches=None):
    # On MPS, evaluate on CPU to avoid buffer crashes
    use_cpu = (device.type == "mps")
    if use_cpu:
        model_eval = model.cpu()
        sector_emb_eval = sector_emb.cpu()
    else:
        model_eval = model
        sector_emb_eval = sector_emb

    model_eval.eval()
    correct_s = correct_c = total = 0
    total_loss = 0
    n_b = 0
    for batch in loader:
        if max_batches and n_b >= max_batches:
            break
        if use_cpu:
            ce, cm, df, cv, cmask, cin, lbl = [t.cpu() for t in batch]
        else:
            ce, cm, df, cv, cmask, cin, lbl = [t.to(device, non_blocking=True) for t in batch]
        with amp_autocast():
            sl, cl, _, _, _ = model_eval(ce, cm.bool(), df, cv, cmask, cin, sector_emb_eval, training=False)
        correct_s += (sl.argmax(-1) == lbl).sum().item()
        correct_c += (cl.argmax(-1) == lbl).sum().item()
        total += lbl.size(0)
        total_loss += F.cross_entropy(sl.float(), lbl, reduction='sum').item()
        n_b += 1

    if use_cpu:
        model.to(device)  # move back to MPS
    model.train()
    return (correct_s/max(total,1)*100, correct_c/max(total,1)*100,
            total_loss/max(total,1), total)

# ============================================================
# PHASE 3: TRAINING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: FAST TRAINING")
print(f"Batch={cfg.batch_size}, AMP={'CUDA' if torch.cuda.is_available() else 'Off'}, Workers={4 if torch.cuda.is_available() else 0}")
print("=" * 60)

model = SchemaV1Production(cfg, N_DS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = GradScaler() if torch.cuda.is_available() else None
ewc = EWC(model, cfg)

# Resume
start_epoch = 0
best_acc = 0
global_step = 0
backbone_frozen = False
samples_seen = 0

if CHECKPOINT_PATH.exists():
    print(f"Resuming from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_accuracy", ckpt.get("accuracy", 0))
    global_step = ckpt.get("step", 0)
    backbone_frozen = ckpt.get("backbone_frozen", False)
    samples_seen = ckpt.get("samples_seen", global_step * cfg.batch_size)  # compat with old checkpoint

    if backbone_frozen:
        model.freeze_backbone()

    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except:
            print("  Fresh optimizer (state mismatch)")

    print(f"  epoch={start_epoch}, step={global_step:,}, best_acc={best_acc:.1f}%, frozen={backbone_frozen}")

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_trainable:,}")

steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * cfg.total_epochs
warmup_steps = steps_per_epoch * cfg.warmup_epochs
print(f"Steps/epoch: {steps_per_epoch:,}, Total: {total_steps:,}")

# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(start_epoch, cfg.total_epochs):
    model.train()
    ep_loss = ep_sloss = ep_closs = ep_mcm = ep_mir = ep_mid = 0
    ep_correct_s = ep_correct_c = ep_n = 0
    t_ep = time.time()

    for bi, batch in enumerate(train_loader):
        ce, cm, df, cv, cmask, cin, lbl = [t.to(device, non_blocking=True) for t in batch]
        cm = cm.bool()
        ce, cm, df, cv, cmask, cin = gpu_augment(ce, cm, df, cv, cmask, cin)

        # Freeze check
        samples_seen += cfg.batch_size
        if not backbone_frozen and samples_seen >= cfg.backbone_freeze_after:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.lr, weight_decay=cfg.weight_decay)
            scaler = GradScaler() if torch.cuda.is_available() else None
            backbone_frozen = True

        lr = get_lr(global_step, warmup_steps, total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Grad accumulation: zero grad only on first micro-batch
        if bi % cfg.grad_accum == 0:
            optimizer.zero_grad(set_to_none=True)

        with amp_autocast():
            sl, cl, mcm_l, mir_l, mid_l = model(ce, cm, df, cv, cmask, cin, sector_emb_matrix, training=True)
            s_loss = F.cross_entropy(sl, lbl, label_smoothing=cfg.label_smoothing)
            c_loss = F.cross_entropy(cl, lbl, label_smoothing=cfg.label_smoothing)
            loss = c_loss + s_loss + cfg.mcm_weight*mcm_l + cfg.miras_weight*mir_l + cfg.midas_weight*mid_l
            loss = loss / cfg.grad_accum  # normalize for accumulation

            if ewc.fisher and global_step % 100 == 0:
                loss = loss + ewc.penalty() / cfg.grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (bi + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        global_step += 1
        bs = lbl.size(0)
        ep_loss += loss.item() * cfg.grad_accum * bs  # undo normalization for logging
        ep_sloss += s_loss.item() * bs
        ep_closs += c_loss.item() * bs
        ep_mcm += mcm_l.item() * bs
        ep_mir += (mir_l.item() if isinstance(mir_l, torch.Tensor) else mir_l) * bs
        ep_mid += mid_l.item() * bs
        with torch.no_grad():
            ep_correct_s += (sl.argmax(-1) == lbl).sum().item()
            ep_correct_c += (cl.argmax(-1) == lbl).sum().item()
        ep_n += bs

        # Log
        if (bi + 1) % cfg.log_every == 0:
            elapsed = time.time() - t_ep
            done = (bi + 1) * cfg.batch_size
            rate = done / elapsed
            eta = (len(train_indices) - done) / max(rate, 1)
            print(f"  E{epoch+1} [{done:,}/{len(train_indices):,}] "
                  f"loss={ep_loss/ep_n:.4f} s_acc={ep_correct_s/ep_n*100:.1f}% c_acc={ep_correct_c/ep_n*100:.1f}% "
                  f"lr={lr:.6f} rate={rate:.0f}/s eta={eta:.0f}s")

        # Val check
        if (bi + 1) % cfg.acc_every == 0:
            mb = cfg.val_samples // (cfg.batch_size * 2) + 1
            sa, ca, vl, vt = evaluate(model, test_loader, sector_emb_matrix, max_batches=mb)
            print(f"    VAL [{vt}] s_acc={sa:.1f}% c_acc={ca:.1f}% loss={vl:.4f}")
            model.train()

        # Checkpoint
        if (bi + 1) % cfg.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else {},
                "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                "best_accuracy": best_acc, "epoch": epoch, "step": global_step,
                "samples_seen": samples_seen, "backbone_frozen": backbone_frozen,
                "dataset_sectors": DS_SECTORS,
                "architecture": "SchemaV1 Production ~122M FAST",
            }, CHECKPOINT_PATH)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Epoch end
    elapsed = time.time() - t_ep
    print(f"\n  Epoch {epoch+1}/{cfg.total_epochs} [{elapsed:.0f}s, {ep_n/elapsed:.0f}/s]")
    print(f"    loss={ep_loss/ep_n:.4f} sector={ep_sloss/ep_n:.3f} cls={ep_closs/ep_n:.3f} "
          f"mcm={ep_mcm/ep_n:.3f} miras={ep_mir/ep_n:.4f} midas={ep_mid/ep_n:.3f}")
    print(f"    train: s_acc={ep_correct_s/ep_n*100:.1f}% c_acc={ep_correct_c/ep_n*100:.1f}%")

    if epoch == 0:
        ewc.register()
        ewc.compute_fisher(train_loader, sector_emb_matrix, n_batches=50)

    # Full val
    print("  Full validation...")
    sa, ca, vl, vt = evaluate(model, test_loader, sector_emb_matrix)
    print(f"  VAL [{vt:,}] s_acc={sa:.1f}% c_acc={ca:.1f}% loss={vl:.4f} (best={max(best_acc,sa):.1f}%)")

    save_d = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else {},
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "accuracy": sa, "best_accuracy": max(best_acc, sa),
        "epoch": epoch+1, "step": global_step,
        "samples_seen": samples_seen, "backbone_frozen": backbone_frozen,
        "dataset_sectors": DS_SECTORS,
        "architecture": "SchemaV1 Production ~122M FAST",
    }
    torch.save(save_d, CHECKPOINT_PATH)

    prev_best = best_acc
    if sa >= best_acc:
        best_acc = sa
        torch.save(save_d, CHECKPOINT_PATH.parent / "schema_v1_production_best.pt")
        print(f"  NEW BEST: {best_acc:.1f}%")

    if not hasattr(model, "_no_improve"):
        model._no_improve = 0
    if sa <= prev_best and epoch > 5:
        model._no_improve += 1
        if model._no_improve >= 5:
            print(f"  Early stop (best={best_acc:.1f}%)")
            break
    else:
        model._no_improve = 0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"DONE — Best: {best_acc:.1f}%, Params: {n_params:,}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"{'='*60}")