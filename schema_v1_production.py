#!/usr/bin/env python3
"""
SchemaLabs V1 Production Training
100M params - Full V0 architecture at full power
MIDAS(10iter) + CellProc(SBERT+fp+sinusoidal) + SchemaProc(SBERT+4layer)
+ LocalReason(4layer axial) + GlobalReason(4layer perceiver, 128 latent)
+ SectorHead(SBERT cosine) + ClassHead + MCM + MIRAS(12 feature) + EWC
"""
import json, os, math, random, time, gc
import numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG - 100M PARAMS
# ============================================================
class Config:
    d_model = 640
    n_heads = 16
    n_latent = 128
    n_layers = 6          # multi-layer for local/global/schema
    dropout = 0.1
    sbert_dim = 384
    fingerprint_dim = 7
    
    max_cols = 30
    max_rows = 10
    
    # Training
    lr = 1e-4
    warmup_epochs = 3
    total_epochs = 20     # 500K data, 20 epoch yeterli
    label_smoothing = 0.1
    weight_decay = 0.01
    grad_accum = 8        # effective batch = 8
    max_grad_norm = 1.0
    
    # Augmentation
    aug_column_shuffle = True
    aug_cell_noise = 0.1
    aug_column_dropout = 0.3
    
    # MIDAS
    midas_iterations = 10
    midas_weight = 0.1
    
    # MCM
    mcm_mask_ratio = 0.15
    mcm_weight = 0.1
    
    # MIRAS (12 active features)
    miras_weight = 0.05
    miras_lq_q = 4
    miras_low_rank_k = 64
    
    # EWC
    ewc_lambda = 1000
    
    # Training strategy
    backbone_freeze_after = 250000  # first 250K: full train, after: frozen backbone
    
    # Paths
    sector_list_path = "data/sector_list_10000.json"
    sector_emb_path = "data/sector_embeddings_10000.npy"
    data_path = "data/v1_training_data.json"
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

cfg = Config()
device = torch.device(cfg.device)
print(f"Device: {device}")

# ============================================================
# PATHS & DATA LOADING
# ============================================================
BASE = Path(os.path.expanduser("~/Desktop/schemalabsai")) if Path(os.path.expanduser("~/Desktop/schemalabsai")).exists() else Path("/opt/schemalabsai")
CHECKPOINT_PATH = BASE / "checkpoints" / "schema_v1_production.pt"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading data...")
t0 = time.time()
with open(BASE / cfg.data_path) as f:
    ALL_DATA = json.load(f)
print(f"Loaded {len(ALL_DATA):,} datasets in {time.time()-t0:.1f}s")

with open(BASE / cfg.sector_list_path) as f:
    sector_data = json.load(f)
    ALL_SECTORS = sector_data["sectors"]
    HIERARCHY = sector_data["hierarchy"]

SECTOR_EMBS = np.load(BASE / cfg.sector_emb_path)

# Build mappings
SUB_TO_MAIN = {}
for main, subs in HIERARCHY.items():
    SUB_TO_MAIN[main] = main
    for s in subs:
        SUB_TO_MAIN[s] = main

MAIN_SECTORS = sorted(HIERARCHY.keys())
N_MAIN = len(MAIN_SECTORS)
MAIN_S2I = {s: i for i, s in enumerate(MAIN_SECTORS)}

# Map dataset sectors
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
N_DS = len(DS_SECTORS)

print(f"Dataset sectors: {N_DS}, Main sectors: {N_MAIN}")
for s in DS_SECTORS:
    c = sum(1 for d in ALL_DATA if d["main_sector"] == s)
    print(f"  {s:25s}: {c:,}")

# ============================================================
# SBERT
# ============================================================
print("Loading SBERT...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

print("Pre-computing column embeddings...")
all_col_names = set()
for d in ALL_DATA:
    for c in d["columns"]:
        all_col_names.add(c.lower().replace("_", " "))
all_col_names = sorted(all_col_names)

# Batch encode
BATCH = 512
COL_EMB_MAP = {}
for i in range(0, len(all_col_names), BATCH):
    batch = all_col_names[i:i+BATCH]
    embs = sbert.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    for name, emb in zip(batch, embs):
        COL_EMB_MAP[name] = emb
    if (i // BATCH) % 10 == 0:
        print(f"  Embedded {min(i+BATCH, len(all_col_names)):,}/{len(all_col_names):,}")
print(f"Embedded {len(COL_EMB_MAP):,} unique columns")

# Main sector SBERT embeddings
MAIN_SECTOR_EMBS = {}
for main in MAIN_SECTORS:
    indices = [i for i, s in enumerate(ALL_SECTORS) if SUB_TO_MAIN.get(s) == main]
    if indices:
        MAIN_SECTOR_EMBS[main] = np.mean(SECTOR_EMBS[indices], axis=0)
    else:
        MAIN_SECTOR_EMBS[main] = sbert.encode([main], convert_to_numpy=True)[0]

DS_SECTOR_MATRIX = np.array([MAIN_SECTOR_EMBS.get(s, sbert.encode([s], convert_to_numpy=True)[0]) for s in DS_SECTORS])
DS_SECTOR_TENSOR = torch.tensor(DS_SECTOR_MATRIX, dtype=torch.float32).to(device)

def get_col_embedding(col_name):
    key = col_name.lower().replace("_", " ")
    if key in COL_EMB_MAP:
        return COL_EMB_MAP[key]
    return sbert.encode([key], convert_to_numpy=True)[0]

# ============================================================
# DATA ENCODING (same as PoC)
# ============================================================
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
        fp = [0,0,0,0,0, unique/max(n_total,1), float(n_total)]
    return [math.copysign(math.log1p(abs(v)),v)/20.0 if abs(v)>1 else v for v in fp]

def encode_dataset(d, augment=False):
    columns = list(d["columns"][:cfg.max_cols])
    all_rows = d.get("sample_rows", [])
    n_cols = len(columns)
    
    if augment and len(all_rows) > cfg.max_rows:
        rows = [all_rows[i] for i in sorted(random.sample(range(len(all_rows)), cfg.max_rows))]
    else:
        rows = all_rows[:cfg.max_rows]
    
    if augment and cfg.aug_column_dropout and n_cols > 4:
        keep = max(3, int(n_cols * (1 - random.uniform(0, cfg.aug_column_dropout))))
        keep_idx = sorted(random.sample(range(n_cols), keep))
        columns = [columns[i] for i in keep_idx]
        rows = [[row[i] if i < len(row) else "" for i in keep_idx] for row in rows]
        n_cols = len(columns)
    
    if augment and n_cols > 2:
        perm = list(range(n_cols))
        random.shuffle(perm)
        columns = [columns[p] for p in perm]
        rows = [[row[p] if p < len(row) else "" for p in perm] for row in rows]
    
    col_embs = torch.zeros(cfg.max_cols, cfg.sbert_dim)
    col_mask = torch.zeros(cfg.max_cols, dtype=torch.bool)
    for i, col in enumerate(columns[:cfg.max_cols]):
        col_embs[i] = torch.tensor(get_col_embedding(col))
        col_mask[i] = True
    
    dist_fps = torch.zeros(cfg.max_cols, cfg.fingerprint_dim)
    cell_values = torch.zeros(cfg.max_rows, cfg.max_cols)
    cell_mask = torch.zeros(cfg.max_rows, cfg.max_cols)
    cell_is_numeric = torch.zeros(cfg.max_rows, cfg.max_cols)
    
    for c_idx in range(min(n_cols, cfg.max_cols)):
        col_vals = [row[c_idx] if c_idx < len(row) else "" for row in rows]
        if augment and cfg.aug_cell_noise:
            noised = []
            for v in col_vals:
                if is_numeric(v):
                    nv = parse_numeric(v) * (1 + random.uniform(-cfg.aug_cell_noise, cfg.aug_cell_noise))
                    noised.append(str(round(nv, 4)))
                else:
                    noised.append(v)
            col_vals = noised
        
        dist_fps[c_idx] = torch.tensor(compute_fingerprint(col_vals))
        for r_idx, v in enumerate(col_vals[:cfg.max_rows]):
            if v and str(v).strip():
                cell_mask[r_idx, c_idx] = 1
                if is_numeric(v):
                    val = parse_numeric(v)
                    cell_values[r_idx, c_idx] = math.copysign(math.log1p(abs(val)), val) / 20.0
                    cell_is_numeric[r_idx, c_idx] = 1
    
    return col_embs, col_mask, dist_fps, cell_values, cell_mask, cell_is_numeric

# ============================================================
# MIDAS - Full Power (10 iterations)
# ============================================================
class MIDAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.imputer = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d*4, d*2), nn.GELU(),
            nn.Linear(d*2, d)
        )
        self.denoiser = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, 1)
        )
        self.norm = nn.LayerNorm(d)
        self.iterations = cfg.midas_iterations
    
    def forward(self, x, cell_mask):
        B, R, C, d = x.shape
        mask_bool = cell_mask.unsqueeze(-1).bool().expand_as(x)
        for _ in range(self.iterations):
            imputed = self.imputer(x)
            x = torch.where(mask_bool, x, imputed)
        x = self.norm(x)
        recon = self.denoiser(x).squeeze(-1)
        return x, recon

# ============================================================
# CellProcessing (512d, full)
# ============================================================
class CellProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.value_proj = nn.Linear(1, d)
        self.numeric_embed = nn.Embedding(2, d)
        self.fp_proj = nn.Linear(cfg.fingerprint_dim, d)
        self.conv1d = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.fusion = nn.Linear(d * 3, d)
        self.norm = nn.LayerNorm(d)
        self.d_model = d
    
    def sinusoidal_position(self, n_pos, d_model, dev):
        pe = torch.zeros(n_pos, d_model, device=dev)
        pos = torch.arange(0, n_pos, dtype=torch.float, device=dev).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=dev) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe
    
    def forward(self, cell_values, cell_is_numeric, dist_fps, col_mask):
        B, R, C = cell_values.shape
        d = self.d_model
        val_emb = self.value_proj(cell_values.unsqueeze(-1))
        type_emb = self.numeric_embed(cell_is_numeric.long())
        fp_emb = self.fp_proj(dist_fps).unsqueeze(1).expand(B, R, C, d)
        pos = self.sinusoidal_position(C, d, cell_values.device).unsqueeze(0).unsqueeze(0).expand(B, R, -1, -1)
        fused = self.fusion(torch.cat([val_emb + pos, type_emb, fp_emb], dim=-1))
        fused = self.norm(fused)
        fused = fused * col_mask.unsqueeze(1).unsqueeze(-1).float()
        return fused

# ============================================================
# SchemaProcessing (4-layer transformer)
# ============================================================
class SchemaProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Linear(cfg.sbert_dim, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d*4,
            batch_first=True, dropout=cfg.dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)
    
    def forward(self, col_embs, col_mask):
        x = self.proj(col_embs)
        x = self.transformer(x, src_key_padding_mask=~col_mask)
        x = self.norm(x) * col_mask.unsqueeze(-1).float()
        return x

# ============================================================
# LocalReasoning (4-layer axial attention)
# ============================================================
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
        # Row-wise
        xr = x.reshape(B*R, C, d)
        mr = (~col_mask).unsqueeze(1).expand(B, R, C).reshape(B*R, C)
        a1, _ = self.row_attn(xr, xr, xr, key_padding_mask=mr)
        x = x + self.norm1(a1.view(B, R, C, d))
        # Column-wise
        xc = x.permute(0, 2, 1, 3).reshape(B*C, R, d)
        a2, _ = self.col_attn(xc, xc, xc)
        x = x + self.norm2(a2.view(B, C, R, d).permute(0, 2, 1, 3))
        # FFN
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

# ============================================================
# GlobalReasoning (4-layer perceiver)
# ============================================================
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

# ============================================================
# SectorHead (SBERT cosine)
# ============================================================
class SectorHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.sbert_dim)
        )
    
    def forward(self, global_repr, schema_pool, sector_emb_matrix):
        combined = torch.cat([global_repr, schema_pool], dim=-1)
        projected = F.normalize(self.proj(combined), dim=-1)
        sector_emb = F.normalize(sector_emb_matrix, dim=-1)
        return torch.mm(projected, sector_emb.t()) * 10

class ClassificationHead(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, n_classes)
        )
    
    def forward(self, x):
        return self.head(x)

# ============================================================
# MCM
# ============================================================
class MCM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d*2, 1)
        )
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

# ============================================================
# MIRAS - 12 Active Features (Full V0)
# ============================================================
class MIRAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        k = cfg.miras_low_rank_k
        
        # 1. HuberBias
        self.huber_bias = nn.Parameter(torch.zeros(d))
        self.huber_delta = 1.0
        
        # 2. LqRetention (q=4)
        self.retention_gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        
        # 3. GDWithMomentum
        self.momentum_beta = 0.9
        self.gd_lr = nn.Parameter(torch.tensor(0.01))
        
        # 4-6. Channel-wise params (η, δ, α)
        self.eta = nn.Parameter(torch.ones(d))
        self.delta_param = nn.Parameter(torch.zeros(d))
        self.alpha = nn.Parameter(torch.ones(d) * 0.5)
        
        # 7-8. Low-rank projection
        self.low_rank_down = nn.Linear(d, k, bias=False)
        self.low_rank_up = nn.Linear(k, d, bias=False)
        
        # 9. Gated output
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        
        # 10. L2 norm
        self.l2_weight = nn.Parameter(torch.tensor(0.001))
        
        # 11. RMSNorm
        self.rms_norm = nn.LayerNorm(d)  # RMSNorm equivalent
        
        # 12. Residual connection (implicit)
    
    def forward(self, x):
        residual = x
        
        # 3. GDWithMomentum - gradient estimate (detached)
        if not hasattr(self, '_momentum_buffer') or self._momentum_buffer is None or self._momentum_buffer.shape != x.shape:
            self._momentum_buffer = torch.zeros_like(x).detach()
        grad_est = (x - residual).detach()
        self._momentum_buffer = (self.momentum_beta * self._momentum_buffer + (1 - self.momentum_beta) * grad_est).detach()
        x = x - self.gd_lr * self._momentum_buffer
        
        # 1. HuberBias
        diff = x - self.huber_bias
        huber = torch.where(diff.abs() <= self.huber_delta, 0.5 * diff**2, self.huber_delta * (diff.abs() - 0.5 * self.huber_delta))
        x = x - 0.01 * huber.sign() * huber.abs().clamp(max=1.0)
        
        # 2. LqRetention
        gate = self.retention_gate(x)
        x = x * gate
        
        # 4-6. Channel-wise
        x = self.eta * x + self.delta_param
        
        # 7-8. Low-rank projection
        low = self.low_rank_up(self.low_rank_down(x))
        x = self.alpha * x + (1 - self.alpha) * low
        
        # 9. Gated output
        gate_out = self.gate(torch.cat([x, residual], dim=-1))
        x = gate_out * x + (1 - gate_out) * residual
        
        # 11. RMSNorm
        x = self.rms_norm(x)
        
        return x
    
    def get_loss(self, x):
        retention = self.retention_gate(x)
        entropy = -(retention * torch.log(retention + 1e-8) + (1-retention) * torch.log(1-retention + 1e-8))
        l2_loss = self.l2_weight * (self.eta**2).mean()
        return -entropy.mean() * 0.01 + l2_loss

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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
    
    def compute_fisher(self, data_batch, encode_fn, sector_emb):
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for d in data_batch[:200]:
            self.model.zero_grad()
            inputs = encode_fn(d)
            ce, cm, df, cv, cmk, cin = [t.unsqueeze(0).to(device) for t in inputs]
            label = torch.tensor([DS_S2I[d["main_sector"]]]).to(device)
            sl, cl, _, _, _ = self.model(ce, cm, df, cv, cmk, cin, sector_emb, training=True)
            loss = F.cross_entropy(sl, label)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data**2
        for n in self.fisher:
            self.fisher[n] /= min(200, len(data_batch))
    
    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.params and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.lam * loss

# ============================================================
# FULL V1 MODEL - 100M PARAMS
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
        
        mcm_loss = torch.tensor(0.0, device=col_embs.device)
        if training:
            cells_masked, mcm_mask = self.mcm.apply_mask(cells, col_mask)
            local_input = cells_masked
        else:
            local_input = cells
            mcm_mask = None
        
        local_out = self.local_reason(local_input, col_mask)
        
        B, R, C, d = local_out.shape
        miras_in = local_out.reshape(B, R*C, d)
        miras_out = self.miras(miras_in)
        local_out = miras_out.reshape(B, R, C, d)
        
        global_repr = self.global_reason(local_out, col_mask)
        schema_pool = (schema * col_mask.unsqueeze(-1).float()).sum(1) / col_mask.sum(1, keepdim=True).float().clamp(min=1)
        
        sector_logits = self.sector_head(global_repr, schema_pool, sector_emb_matrix)
        cls_logits = self.cls_head(global_repr)
        
        if training and mcm_mask is not None:
            mcm_loss = self.mcm.predict(local_out, mcm_mask, cell_values.unsqueeze(-1).expand_as(local_out)[..., 0])
        
        miras_loss = self.miras.get_loss(miras_in) if training else torch.tensor(0.0, device=col_embs.device)
        midas_loss = F.mse_loss(midas_recon, cell_values) if training else torch.tensor(0.0, device=col_embs.device)
        
        return sector_logits, cls_logits, mcm_loss, miras_loss, midas_loss
    
    def freeze_backbone(self):
        """Freeze everything except heads"""
        for name, param in self.named_parameters():
            if "sector_head" not in name and "cls_head" not in name:
                param.requires_grad = False
        print("  Backbone frozen - only heads training")
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

# ============================================================
# TRAINING
# ============================================================
def get_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

print("\n" + "=" * 60)
print("SchemaLabs V1 Production Training")
print(f"Target: ~100M params")
print("=" * 60)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Split
random.shuffle(ALL_DATA)
split = int(0.95 * len(ALL_DATA))
train_data = ALL_DATA[:split]
test_data = ALL_DATA[split:]
print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")

# Model
model = SchemaV1Production(cfg, N_DS).to(device)

# Resume from checkpoint if exists
start_epoch = 0
if CHECKPOINT_PATH.exists():
    print(f"Resuming from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("accuracy", 0)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt.get("step", 0)
    backbone_frozen = ckpt.get("backbone_frozen", False)
    if backbone_frozen:
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except:
                pass
    mid_epoch_di = ckpt.get("mid_epoch_di", 0)
    print(f"  Resumed from epoch {start_epoch}, step={step:,}, mid_di={mid_epoch_di:,}, best_acc={best_acc:.1f}%, frozen={backbone_frozen}")
else:
    best_acc = 0
    step = 0
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_params:,}")
print(f"Trainable params: {n_trainable:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
ewc = EWC(model, cfg)

# Training loop
total_steps = len(train_data) * cfg.total_epochs
warmup_steps = len(train_data) * cfg.warmup_epochs
step = 0
backbone_frozen = False

for epoch in range(start_epoch, cfg.total_epochs):
    model.train()
    random.shuffle(train_data)
    
    epoch_loss = 0
    epoch_sector = 0
    epoch_cls = 0
    epoch_mcm = 0
    epoch_miras = 0
    epoch_midas = 0
    epoch_samples = 0
    
    t_epoch = time.time()
    
    skip_to = 0
    if epoch == start_epoch and 'mid_epoch_di' in dir() and mid_epoch_di > 0:
        skip_to = mid_epoch_di
        mid_epoch_di = 0  # sadece ilk epoch'ta skip
        print(f"  Skipping to sample {skip_to:,}")
    
    for di, d in enumerate(train_data):
        if di < skip_to:
            step += 1
            continue
        # Backbone freeze strategy
        dataset_idx = epoch * len(train_data) + di
        if not backbone_frozen and dataset_idx >= cfg.backbone_freeze_after:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
            backbone_frozen = True
        
        # LR schedule
        lr = get_lr(step, warmup_steps, total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # Encode
        inputs = encode_dataset(d, augment=True)
        ce, cm, df, cv, cmk, cin = [t.unsqueeze(0).to(device) for t in inputs]
        label = torch.tensor([DS_S2I[d["main_sector"]]]).to(device)
        
        # Forward
        sl, cl, mcm_l, mir_l, mid_l = model(ce, cm, df, cv, cmk, cin, DS_SECTOR_TENSOR, training=True)
        
        sector_loss = F.cross_entropy(sl, label, label_smoothing=cfg.label_smoothing)
        cls_loss = F.cross_entropy(cl, label, label_smoothing=cfg.label_smoothing)
        loss = cls_loss + sector_loss + cfg.mcm_weight * mcm_l + cfg.miras_weight * mir_l + cfg.midas_weight * mid_l
        
        if ewc.fisher and step % 1000 == 0:
            ewc_loss = ewc.penalty()
            loss = loss + ewc_loss.detach()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        
        step += 1
        epoch_loss += loss.item()
        epoch_sector += sector_loss.item()
        epoch_cls += cls_loss.item()
        epoch_mcm += mcm_l.item()
        epoch_miras += mir_l.item() if isinstance(mir_l, torch.Tensor) else mir_l
        epoch_midas += mid_l.item()
        epoch_samples += 1
        
        # Progress
        if (di + 1) % 10000 == 0:
            elapsed = time.time() - t_epoch
            rate = (di + 1) / elapsed
            eta = (len(train_data) - di - 1) / rate
            avg_loss = epoch_loss / epoch_samples
            acc_str = ""
            if (di + 1) % 50000 == 0:
                model.eval()
                mc = 0
                with torch.no_grad():
                    for td in test_data[:500]:
                        inp = encode_dataset(td, augment=False)
                        tc = [t.unsqueeze(0).to(device) for t in inp]
                        sl2, _, _, _, _ = model(*tc, DS_SECTOR_TENSOR, training=False)
                        if sl2.argmax(-1).item() == DS_S2I[td["main_sector"]]:
                            mc += 1
                acc_str = f" acc={mc/5:.1f}%"
                model.train()
            print(f"  Epoch {epoch+1} [{di+1:,}/{len(train_data):,}] loss={avg_loss:.4f} lr={lr:.6f} rate={rate:.0f}/s eta={eta:.0f}s{acc_str}")
        
        # Memory cleanup
        if (di + 1) % 10000 == 0:
            gc.collect()
        
        # Mid-epoch checkpoint her 100K'da
        if (di + 1) % 100000 == 0:
            mid_save = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                "accuracy": best_acc,
                "best_accuracy": best_acc,
                "epoch": epoch,
                "step": step,
                "mid_epoch_di": di + 1,
                "backbone_frozen": backbone_frozen,
                "dataset_sectors": DS_SECTORS,
                "architecture": "SchemaV1 Production ~122M",
            }
            torch.save(mid_save, CHECKPOINT_PATH)
            print(f"  Mid-epoch checkpoint saved (epoch {epoch+1}, sample {di+1:,})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # EWC register after first epoch
    if epoch == 0:
        ewc.register()
        ewc.compute_fisher(train_data, encode_dataset, DS_SECTOR_TENSOR)
    
    # Epoch summary
    n = epoch_samples
    elapsed = time.time() - t_epoch
    print(f"\n  Epoch {epoch+1}/{cfg.total_epochs} [{elapsed:.0f}s]: "
          f"loss={epoch_loss/n:.4f} sector={epoch_sector/n:.3f} cls={epoch_cls/n:.3f} "
          f"mcm={epoch_mcm/n:.3f} miras={epoch_miras/n:.4f} midas={epoch_midas/n:.3f}")
    
    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for d in test_data[:1000]:  # Sample 1000 for speed
            inputs = encode_dataset(d, augment=False)
            ce, cm, df, cv, cmk, cin = [t.unsqueeze(0).to(device) for t in inputs]
            sl, _, _, _, _ = model(ce, cm, df, cv, cmk, cin, DS_SECTOR_TENSOR, training=False)
            pred = sl.argmax(-1).item()
            if pred == DS_S2I[d["main_sector"]]:
                correct += 1
    
    acc = correct / min(len(test_data), 1000) * 100
    print(f"  Val accuracy: {acc:.1f}% (best={max(best_acc, acc):.1f}%)")
    
    # Save every epoch (resume icin)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "accuracy": acc,
        "best_accuracy": max(best_acc, acc),
        "epoch": epoch + 1,
        "step": step,
        "backbone_frozen": backbone_frozen,
        "dataset_sectors": DS_SECTORS,
        "architecture": "SchemaV1 Production ~95M: MIDAS+CellProc+SchemaProc+6xAxialAttn+6xPerceiver+SectorHead+MCM+MIRAS12+EWC",
    }
    
    # Her epoch kaydet (resume icin) - onceki epoch sil
    prev_epoch_path = CHECKPOINT_PATH.parent / f"schema_v1_production_epoch{epoch}.pt"
    if prev_epoch_path.exists():
        prev_epoch_path.unlink()
    torch.save(save_dict, CHECKPOINT_PATH)
    print(f"  Checkpoint saved: {CHECKPOINT_PATH} (epoch {epoch+1}, ~{CHECKPOINT_PATH.stat().st_size/1024/1024:.0f}MB)")
    
    # Best model ayri kaydet
    if acc > best_acc:
        best_acc = acc
        best_path = CHECKPOINT_PATH.parent / "schema_v1_production_best.pt"
        torch.save(save_dict, best_path)
        print(f"  Best model saved: {best_path}")
    
    # Early stop: 3 epoch ust uste iyilesme yoksa dur
    if not hasattr(model, "_no_improve_count"):
        model._no_improve_count = 0
    if acc <= best_acc and epoch > 5:
        model._no_improve_count += 1
        if model._no_improve_count >= 5:
            print(f"  Early stop: 5 epoch no improvement (best={best_acc:.1f}%)")
            break
    else:
        model._no_improve_count = 0

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"Best accuracy: {best_acc:.1f}%")
print(f"Params: {n_params:,}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"{'='*60}")
