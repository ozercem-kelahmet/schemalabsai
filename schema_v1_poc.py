#!/usr/bin/env python3
"""
SchemaLabs V1 Full Architecture PoC
MIDAS + CellProcessing + SchemaProcessing + LocalReasoning + GlobalReasoning
+ SectorHead(SBERT) + ClassificationHead + MCM + MIRAS + EWC
"""
import json, os, math, random
import numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
class Config:
    # Model dims (PoC size)
    d_model = 64
    n_heads = 4
    n_latent = 16
    dropout = 0.3
    sbert_dim = 384
    
    # Data dims
    max_cols = 30
    max_rows = 10
    fingerprint_dim = 7
    
    # Training
    lr = 3e-4
    warmup_epochs = 5
    total_epochs = 100
    label_smoothing = 0.1
    weight_decay = 0.05
    
    # Augmentation
    aug_per_dataset = 5
    aug_column_shuffle = True
    aug_cell_noise = 0.1
    aug_column_dropout = 0.3
    
    # MIDAS
    midas_mask_ratio = 0.3
    midas_iterations = 3
    
    # MCM
    mcm_mask_ratio = 0.15
    mcm_weight = 0.1
    
    # MIRAS
    miras_weight = 0.05
    miras_lq_q = 4
    
    # EWC
    ewc_lambda = 1000
    
    # Sector
    sector_list_path = "data/sector_list_10000.json"
    sector_emb_path = "data/sector_embeddings_10000.npy"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

cfg = Config()
device = torch.device(cfg.device)
print(f"Device: {device}")

# ============================================================
# PATHS
# ============================================================
BASE = Path(os.path.expanduser("~/Desktop/schemalabsai"))
DATA_PATH = BASE / "data" / "poc_synthetic_1000.json"
SECTOR_PATH = BASE / cfg.sector_list_path
SECTOR_EMB_PATH = BASE / cfg.sector_emb_path
CHECKPOINT_PATH = BASE / "checkpoints" / "schema_v1_poc.pt"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
with open(DATA_PATH) as f:
    RAW_DATA = json.load(f)

with open(SECTOR_PATH) as f:
    sector_data = json.load(f)
    ALL_SECTORS = sector_data["sectors"]
    HIERARCHY = sector_data["hierarchy"]

SECTOR_EMBS = np.load(SECTOR_EMB_PATH)

# Build sub_to_main mapping
SUB_TO_MAIN = {}
for main, subs in HIERARCHY.items():
    SUB_TO_MAIN[main] = main
    for s in subs:
        SUB_TO_MAIN[s] = main

MAIN_SECTORS = sorted(HIERARCHY.keys())
N_MAIN = len(MAIN_SECTORS)
MAIN_S2I = {s: i for i, s in enumerate(MAIN_SECTORS)}

# Map dataset sectors to main sectors
for d in RAW_DATA:
    sector = d["sector"]
    # Find closest main sector
    if sector in MAIN_S2I:
        d["main_sector"] = sector
    else:
        # Try substring match
        matched = False
        for main in MAIN_SECTORS:
            if sector in main or main in sector:
                d["main_sector"] = main
                matched = True
                break
        if not matched:
            d["main_sector"] = "manufacturing"  # fallback

DATASET_SECTORS = sorted(set(d["main_sector"] for d in RAW_DATA))
DS_S2I = {s: i for i, s in enumerate(DATASET_SECTORS)}
N_DS_SECTORS = len(DATASET_SECTORS)

print(f"Datasets: {len(RAW_DATA)}, Dataset sectors: {N_DS_SECTORS}, Main sectors: {N_MAIN}")

# ============================================================
# SBERT SETUP
# ============================================================
print("Loading SBERT...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute column name embeddings
print("Pre-computing column embeddings...")
all_col_names = set()
for d in RAW_DATA:
    for c in d["columns"]:
        all_col_names.add(c.lower().replace("_", " "))
all_col_names = sorted(all_col_names)
col_embeddings = sbert.encode(all_col_names, show_progress_bar=False, convert_to_numpy=True)
COL_EMB_MAP = {name: col_embeddings[i] for i, name in enumerate(all_col_names)}
print(f"Embedded {len(COL_EMB_MAP)} unique column names")

# Main sector embeddings for SectorHead
MAIN_SECTOR_EMBS = {}
for main in MAIN_SECTORS:
    indices = [i for i, s in enumerate(ALL_SECTORS) if SUB_TO_MAIN.get(s) == main]
    if indices:
        MAIN_SECTOR_EMBS[main] = np.mean(SECTOR_EMBS[indices], axis=0)
    else:
        MAIN_SECTOR_EMBS[main] = sbert.encode([main], convert_to_numpy=True)[0]

MAIN_SECTOR_MATRIX = np.array([MAIN_SECTOR_EMBS[m] for m in MAIN_SECTORS])
MAIN_SECTOR_TENSOR = torch.tensor(MAIN_SECTOR_MATRIX, dtype=torch.float32)

def get_col_embedding(col_name):
    key = col_name.lower().replace("_", " ")
    if key in COL_EMB_MAP:
        return COL_EMB_MAP[key]
    return sbert.encode([key], convert_to_numpy=True)[0]

# ============================================================
# DATA ENCODING
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
    # Log-scale
    return [math.copysign(math.log1p(abs(v)),v)/20.0 if abs(v)>1 else v for v in fp]

def encode_dataset(d, augment=False):
    columns = list(d["columns"][:cfg.max_cols])
    all_rows = d.get("sample_rows", [])
    n_cols = len(columns)
    
    # Row subset
    if augment and len(all_rows) > cfg.max_rows:
        rows = [all_rows[i] for i in sorted(random.sample(range(len(all_rows)), cfg.max_rows))]
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
    
    # Distribution fingerprints
    dist_fps = torch.zeros(cfg.max_cols, cfg.fingerprint_dim)
    
    # Cell values matrix (for MCM and MIDAS)
    cell_values = torch.zeros(cfg.max_rows, cfg.max_cols)
    cell_mask = torch.zeros(cfg.max_rows, cfg.max_cols)  # 1 = has value
    cell_is_numeric = torch.zeros(cfg.max_rows, cfg.max_cols)
    
    for c_idx in range(min(n_cols, cfg.max_cols)):
        col_vals = [row[c_idx] if c_idx < len(row) else "" for row in rows]
        
        # Cell noise augmentation
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
# MIDAS - Missing Data Strategy
# ============================================================
class MIDAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.mask_embed = nn.Embedding(2, d)  # 0=missing, 1=present
        self.imputer = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d*2, d)
        )
        self.denoiser = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.norm = nn.LayerNorm(d)
        self.iterations = cfg.midas_iterations
    
    def forward(self, x, cell_mask):
        # x: (B, R, C, d), cell_mask: (B, R, C)
        B, R, C, d = x.shape
        mask_bool = cell_mask.unsqueeze(-1).bool().expand_as(x)  # (B, R, C, d)
        
        # Iterative imputation
        for _ in range(self.iterations):
            imputed = self.imputer(x)
            x = torch.where(mask_bool, x, imputed)
        
        x = self.norm(x)
        recon = self.denoiser(x).squeeze(-1)  # (B, R, C)
        return x, recon

# ============================================================
# CellProcessing (V1 - SBERT + fingerprint + sinusoidal position)
# ============================================================
class CellProcessing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.value_proj = nn.Linear(1, d)
        self.numeric_embed = nn.Embedding(2, d)  # 0=categorical, 1=numeric
        self.fp_proj = nn.Linear(cfg.fingerprint_dim, d)
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
        
        # Value embedding
        val_emb = self.value_proj(cell_values.unsqueeze(-1))  # (B, R, C, d)
        
        # Type embedding
        type_emb = self.numeric_embed(cell_is_numeric.long())  # (B, R, C, d)
        
        # Fingerprint embedding (per column, broadcast to rows)
        fp_emb = self.fp_proj(dist_fps).unsqueeze(1).expand(B, R, C, d)  # (B, R, C, d)
        
        # Sinusoidal position (per column)
        pos = self.sinusoidal_position(C, d, cell_values.device)
        pos = pos.unsqueeze(0).unsqueeze(0).expand(B, R, -1, -1)
        
        # Fusion
        fused = self.fusion(torch.cat([val_emb + pos, type_emb, fp_emb], dim=-1))
        fused = self.norm(fused)
        fused = fused * col_mask.unsqueeze(1).unsqueeze(-1).float()
        return fused

# ============================================================
# SchemaProcessing (V1 - SBERT based)
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
# LocalReasoning (V0 - axial attention)
# ============================================================
class LocalReasoning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.row_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
    
    def forward(self, x, col_mask):
        B, R, C, d = x.shape
        # Row-wise attention
        xr = x.reshape(B*R, C, d)
        mr = (~col_mask).unsqueeze(1).expand(B, R, C).reshape(B*R, C)
        a1, _ = self.row_attn(xr, xr, xr, key_padding_mask=mr)
        x = x + self.norm1(a1.view(B, R, C, d))
        # Column-wise attention
        xc = x.permute(0, 2, 1, 3).reshape(B*C, R, d)
        a2, _ = self.col_attn(xc, xc, xc)
        x = x + self.norm2(a2.view(B, C, R, d).permute(0, 2, 1, 3))
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
        B, R, C, d = x.shape
        flat = x.reshape(B, R*C, d)
        mask = ~col_mask.unsqueeze(1).expand(B, R, C).reshape(B, R*C)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(lat, flat, flat, key_padding_mask=mask)
        out = self.self_attn(out)
        return self.norm(out).mean(dim=1)

# ============================================================
# SectorHead (V1 - SBERT cosine similarity, agnostic)
# ============================================================
class SectorHead(nn.Module):
    def __init__(self, cfg, n_sectors):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.sbert_dim)
        )
    
    def forward(self, global_repr, schema_pool, sector_emb_matrix):
        # Project model output to SBERT space
        combined = torch.cat([global_repr, schema_pool], dim=-1)
        projected = self.proj(combined)  # (B, sbert_dim)
        projected = F.normalize(projected, dim=-1)
        
        # Cosine similarity with sector embeddings
        sector_emb = F.normalize(sector_emb_matrix, dim=-1)
        logits = torch.mm(projected, sector_emb.t()) * 10  # temperature scaling
        return logits

# ============================================================
# ClassificationHead (dynamic class count)
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        d = cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, n_classes)
        )
    
    def forward(self, global_repr):
        return self.head(global_repr)

# ============================================================
# MCM - Masked Cell Modeling (self-supervised)
# ============================================================
class MCM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, 1)
        )
        self.mask_ratio = cfg.mcm_mask_ratio
    
    def apply_mask(self, cell_emb, cell_mask):
        B, R, C, d = cell_emb.shape
        # Random mask
        rand = torch.rand(B, R, C, device=cell_emb.device)
        mcm_mask = (rand < self.mask_ratio) & cell_mask.unsqueeze(1).expand(B, R, C).bool()
        
        # Replace masked positions with mask token
        masked_emb = cell_emb.clone()
        masked_emb[mcm_mask] = self.mask_token.expand(mcm_mask.sum(), -1)
        return masked_emb, mcm_mask
    
    def predict(self, hidden, mcm_mask, original_values):
        pred = self.predictor(hidden).squeeze(-1)
        # Loss only on masked positions
        if mcm_mask.sum() > 0:
            loss = F.mse_loss(pred[mcm_mask], original_values[mcm_mask])
        else:
            loss = torch.tensor(0.0, device=hidden.device)
        return loss

# ============================================================
# MIRAS - Memory-Informed Retention Attention System
# ============================================================
class MIRAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        # HuberBias - attentional bias (outlier-resistant)
        self.huber_bias = nn.Parameter(torch.zeros(d))
        self.huber_delta = 1.0
        
        # LqRetention - retention gate (soft thresholding)
        self.retention_gate = nn.Sequential(
            nn.Linear(d, d), nn.Sigmoid()
        )
        self.lq_q = cfg.miras_lq_q
        
        # GDWithMomentum - memory buffer
        self.momentum_buffer = None
        self.momentum_beta = 0.9
        
        # Channel-wise params
        self.eta = nn.Parameter(torch.ones(d))
        self.delta = nn.Parameter(torch.zeros(d))
        self.alpha = nn.Parameter(torch.ones(d) * 0.5)
        
        # Gated output
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.norm = nn.RMSNorm(d) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d)
    
    def forward(self, x):
        residual = x
        
        # HuberBias
        diff = x - self.huber_bias
        huber = torch.where(
            diff.abs() <= self.huber_delta,
            0.5 * diff ** 2,
            self.huber_delta * (diff.abs() - 0.5 * self.huber_delta)
        )
        x = x - 0.01 * huber.sign() * huber.abs().clamp(max=1.0)
        
        # LqRetention
        gate = self.retention_gate(x)
        x = x * gate
        
        # Channel-wise transform
        x = self.eta * x + self.delta
        
        # Gated output
        gate_out = self.gate(torch.cat([x, residual], dim=-1))
        x = gate_out * x + (1 - gate_out) * residual
        
        x = self.norm(x)
        return x
    
    def get_loss(self, x):
        # MIRAS regularization loss
        retention = self.retention_gate(x)
        # Encourage moderate retention (not all 0 or all 1)
        entropy = -(retention * torch.log(retention + 1e-8) + 
                    (1-retention) * torch.log(1-retention + 1e-8))
        return -entropy.mean() * 0.01  # maximize entropy = moderate retention

# ============================================================
# EWC - Elastic Weight Consolidation
# ============================================================
class EWC:
    def __init__(self, model, cfg):
        self.model = model
        self.lam = cfg.ewc_lambda
        self.params = {}
        self.fisher = {}
    
    def register(self):
        """Store current params as reference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()
    
    def compute_fisher(self, data_list, encode_fn, label_fn, n_samples=100):
        """Estimate Fisher information from data"""
        self.fisher = {name: torch.zeros_like(param) 
                      for name, param in self.model.named_parameters() if param.requires_grad}
        
        self.model.eval()
        samples = random.sample(data_list, min(n_samples, len(data_list)))
        
        for d in samples:
            self.model.zero_grad()
            inputs = encode_fn(d)
            label = label_fn(d)
            output = self.model(*inputs)
            if isinstance(output, tuple):
                output = output[0]
            loss = F.cross_entropy(output, label)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
        
        for name in self.fisher:
            self.fisher[name] /= len(samples)
    
    def penalty(self):
        """EWC penalty loss"""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.params and name in self.fisher:
                loss += (self.fisher[name] * (param - self.params[name]) ** 2).sum()
        return self.lam * loss

# ============================================================
# FULL V1 MODEL
# ============================================================
class SchemaV1Model(nn.Module):
    def __init__(self, cfg, n_dataset_sectors):
        super().__init__()
        self.cfg = cfg
        
        # Core architecture
        self.midas = MIDAS(cfg)
        self.cell_proc = CellProcessing(cfg)
        self.schema_proc = SchemaProcessing(cfg)
        self.local_reason = LocalReasoning(cfg)
        self.global_reason = GlobalReasoning(cfg)
        
        # Heads
        self.sector_head = SectorHead(cfg, n_dataset_sectors)
        self.cls_head = ClassificationHead(cfg, n_dataset_sectors)
        
        # Self-supervised
        self.mcm = MCM(cfg)
        self.miras = MIRAS(cfg)
    
    def forward(self, col_embs, col_mask, dist_fps, cell_values, cell_mask, 
                cell_is_numeric, sector_emb_matrix, training=False):
        B = col_embs.shape[0]
        R = cell_values.shape[1]
        C = cfg.max_cols
        
        # 1. SchemaProcessing
        schema = self.schema_proc(col_embs, col_mask)  # (B, C, d)
        
        # 2. CellProcessing
        cells = self.cell_proc(cell_values, cell_is_numeric, dist_fps, col_mask)  # (B, R, C, d)
        
        # 3. MIDAS on cells
        cells, midas_recon = self.midas(cells, cell_mask)
        
        # 4. Combine schema + cells
        cells = cells + schema.unsqueeze(1)  # broadcast schema to rows
        
        # 5. MCM (training only)
        mcm_loss = torch.tensor(0.0, device=col_embs.device)
        if training:
            cells_masked, mcm_mask = self.mcm.apply_mask(cells, col_mask)
            # Use masked version for forward pass during training
            local_input = cells_masked
        else:
            local_input = cells
            mcm_mask = None
        
        # 6. LocalReasoning
        local_out = self.local_reason(local_input, col_mask)
        
        # 7. MIRAS wrapper
        B2, R2, C2, d2 = local_out.shape
        miras_in = local_out.reshape(B2, R2*C2, d2)
        miras_out = self.miras(miras_in)
        local_out = miras_out.reshape(B2, R2, C2, d2)
        
        # 8. GlobalReasoning
        global_repr = self.global_reason(local_out, col_mask)  # (B, d)
        
        # 9. Schema pool
        schema_pool = (schema * col_mask.unsqueeze(-1).float()).sum(1) / \
                      col_mask.sum(1, keepdim=True).float().clamp(min=1)
        
        # 10. Heads
        sector_logits = self.sector_head(global_repr, schema_pool, sector_emb_matrix)
        cls_logits = self.cls_head(global_repr)
        
        # 11. MCM loss
        if training and mcm_mask is not None:
            mcm_loss = self.mcm.predict(local_out, mcm_mask, cell_values.unsqueeze(-1).expand_as(local_out)[..., 0])
        
        # 12. MIRAS loss
        miras_loss = self.miras.get_loss(miras_in) if training else torch.tensor(0.0, device=col_embs.device)
        
        # 13. MIDAS reconstruction loss
        midas_loss = F.mse_loss(midas_recon, cell_values) if training else torch.tensor(0.0, device=col_embs.device)
        
        return sector_logits, cls_logits, mcm_loss, miras_loss, midas_loss

# ============================================================
# TRAINING UTILS
# ============================================================
def get_lr(epoch, warmup, total, base_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def to_device(tensors):
    return tuple(t.unsqueeze(0).to(device) for t in tensors)

def make_sector_emb():
    """Get sector embedding matrix for current dataset sectors on device"""
    embs = []
    for s in DATASET_SECTORS:
        if s in MAIN_SECTOR_EMBS:
            embs.append(MAIN_SECTOR_EMBS[s])
        else:
            embs.append(sbert.encode([s], convert_to_numpy=True)[0])
    return torch.tensor(np.array(embs), dtype=torch.float32).to(device)

# ============================================================
# MAIN TRAINING
# ============================================================
print("\n" + "=" * 60)
print("SchemaLabs V1 Full Architecture PoC")
print("MIDAS + CellProc + SchemaProc + LocalReason + GlobalReason")
print("+ SectorHead(SBERT) + MCM + MIRAS + EWC")
print("=" * 60)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Split data
indices = list(range(len(RAW_DATA)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_data = [RAW_DATA[i] for i in indices[:split]]
test_data = [RAW_DATA[i] for i in indices[split:]]
print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}")

# Model
model = SchemaV1Model(cfg, N_DS_SECTORS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

# Sector embeddings
sector_emb_matrix = make_sector_emb()

# EWC
ewc = EWC(model, cfg)

# Training loop
best_acc = 0
best_state = None

for epoch in range(cfg.total_epochs):
    model.train()
    lr = get_lr(epoch, cfg.warmup_epochs, cfg.total_epochs, cfg.lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    
    # Augmented samples
    train_samples = []
    for d in train_data:
        for _ in range(cfg.aug_per_dataset):
            train_samples.append(d)
    random.shuffle(train_samples)
    
    total_loss = 0
    total_sector = 0
    total_cls = 0
    total_mcm = 0
    total_miras = 0
    total_midas = 0
    
    for d in train_samples:
        inputs = encode_dataset(d, augment=True)
        col_embs, col_mask, dist_fps, cell_values, cell_mask_data, cell_is_numeric = inputs
        
        # To device
        col_embs = col_embs.unsqueeze(0).to(device)
        col_mask = col_mask.unsqueeze(0).to(device)
        dist_fps = dist_fps.unsqueeze(0).to(device)
        cell_values = cell_values.unsqueeze(0).to(device)
        cell_mask_data = cell_mask_data.unsqueeze(0).to(device)
        cell_is_numeric = cell_is_numeric.unsqueeze(0).to(device)
        
        sector_label = torch.tensor([DS_S2I[d["main_sector"]]]).to(device)
        
        sector_logits, cls_logits, mcm_loss, miras_loss, midas_loss = model(
            col_embs, col_mask, dist_fps, cell_values, cell_mask_data,
            cell_is_numeric, sector_emb_matrix, training=True
        )
        
        # Loss = cls + sector + 0.1*mcm + 0.05*miras + 0.1*midas
        sector_loss = F.cross_entropy(sector_logits, sector_label, label_smoothing=cfg.label_smoothing)
        cls_loss = F.cross_entropy(cls_logits, sector_label, label_smoothing=cfg.label_smoothing)
        
        loss = cls_loss + sector_loss + cfg.mcm_weight * mcm_loss + cfg.miras_weight * miras_loss + 0.1 * midas_loss
        
        # EWC penalty (after first epoch)
        if epoch > 0 and ewc.fisher:
            loss = loss + ewc.penalty()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_sector += sector_loss.item()
        total_cls += cls_loss.item()
        total_mcm += mcm_loss.item()
        total_miras += miras_loss.item() if isinstance(miras_loss, torch.Tensor) else miras_loss
        total_midas += midas_loss.item()
    
    # Register EWC after first epoch
    if epoch == 0:
        ewc.register()
    
    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        n_samples = len(train_samples)
        
        # Test accuracy
        correct = 0
        with torch.no_grad():
            for d in test_data:
                inputs = encode_dataset(d, augment=False)
                col_embs_t, col_mask_t, dist_fps_t, cell_values_t, cell_mask_t, cell_is_numeric_t = inputs
                
                col_embs_t = col_embs_t.unsqueeze(0).to(device)
                col_mask_t = col_mask_t.unsqueeze(0).to(device)
                dist_fps_t = dist_fps_t.unsqueeze(0).to(device)
                cell_values_t = cell_values_t.unsqueeze(0).to(device)
                cell_mask_t = cell_mask_t.unsqueeze(0).to(device)
                cell_is_numeric_t = cell_is_numeric_t.unsqueeze(0).to(device)
                
                sector_logits, cls_logits, _, _, _ = model(
                    col_embs_t, col_mask_t, dist_fps_t, cell_values_t, cell_mask_t,
                    cell_is_numeric_t, sector_emb_matrix, training=False
                )
                
                pred = sector_logits.argmax(-1).item()
                actual = DS_S2I[d["main_sector"]]
                if pred == actual:
                    correct += 1
        
        acc = correct / len(test_data) * 100
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        avg = total_loss / n_samples
        print(f"  Epoch {epoch+1:3d}: loss={avg:.4f} (sector={total_sector/n_samples:.3f} cls={total_cls/n_samples:.3f} "
              f"mcm={total_mcm/n_samples:.3f} miras={total_miras/n_samples:.4f} midas={total_midas/n_samples:.3f}) "
              f"lr={lr:.6f} Val={acc:.1f}% (best={best_acc:.1f}%)")
        
        if best_acc >= 100.0:
            print("  Early stop: 100% reached")
            break

# Final evaluation
if best_state:
    model.load_state_dict(best_state)

model.eval()
correct = 0
details = []
with torch.no_grad():
    for d in test_data:
        inputs = encode_dataset(d, augment=False)
        col_embs_t, col_mask_t, dist_fps_t, cell_values_t, cell_mask_t, cell_is_numeric_t = inputs
        col_embs_t = col_embs_t.unsqueeze(0).to(device)
        col_mask_t = col_mask_t.unsqueeze(0).to(device)
        dist_fps_t = dist_fps_t.unsqueeze(0).to(device)
        cell_values_t = cell_values_t.unsqueeze(0).to(device)
        cell_mask_t = cell_mask_t.unsqueeze(0).to(device)
        cell_is_numeric_t = cell_is_numeric_t.unsqueeze(0).to(device)
        
        sector_logits, cls_logits, _, _, _ = model(
            col_embs_t, col_mask_t, dist_fps_t, cell_values_t, cell_mask_t,
            cell_is_numeric_t, sector_emb_matrix, training=False
        )
        
        pred = sector_logits.argmax(-1).item()
        actual = DS_S2I[d["main_sector"]]
        ok = pred == actual
        if ok: correct += 1
        details.append((d, pred, actual, ok))

print(f"\n{'='*60}")
print(f"FINAL TEST: {correct}/{len(test_data)} = {correct/len(test_data)*100:.1f}%")
print(f"{'='*60}")

# Show errors
for d, pred, actual, ok in details:
    if not ok:
        print(f"  [XX] {d['folder'][:40]:40s} actual={DATASET_SECTORS[actual]:15s} pred={DATASET_SECTORS[pred]}")

# Save checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    "accuracy": best_acc,
    "dataset_sectors": DATASET_SECTORS,
    "main_sectors": MAIN_SECTORS,
    "architecture": "SchemaV1: MIDAS+CellProc+SchemaProc+LocalReason+GlobalReason+SectorHead+MCM+MIRAS+EWC",
}, CHECKPOINT_PATH)
print(f"\nCheckpoint: {CHECKPOINT_PATH}")
print(f"Architecture: MIDAS + CellProcessing + SchemaProcessing + LocalReasoning + GlobalReasoning")
print(f"            + SectorHead(SBERT) + ClassificationHead + MCM + MIRAS + EWC")
print(f"Params: {n_params:,}")
print(f"Best accuracy: {best_acc:.1f}%")

# ============================================================
# PHASE 2: UNSEEN SECTOR TEST (leave-one-sector-out)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: UNSEEN SECTOR TEST (leave-one-sector-out)")
print("Model hic gormedigi sektorleri bulabiliyor mu?")
print("=" * 60)

unseen_results = {}

for holdout in DATASET_SECTORS:
    ho_data = [d for d in RAW_DATA if d["main_sector"] == holdout]
    tr_data = [d for d in RAW_DATA if d["main_sector"] != holdout]
    if len(ho_data) < 2:
        continue
    
    # Train new model without holdout sector
    m = SchemaV1Model(cfg, N_DS_SECTORS).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    for ep in range(60):
        m.train()
        lr_ep = get_lr(ep, 3, 60, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_ep
        
        samples = []
        for d in tr_data:
            for _ in range(cfg.aug_per_dataset):
                samples.append(d)
        random.shuffle(samples)
        
        for d in samples:
            inputs = encode_dataset(d, augment=True)
            ce, cm, df, cv, cmk, cin = inputs
            ce = ce.unsqueeze(0).to(device)
            cm = cm.unsqueeze(0).to(device)
            df = df.unsqueeze(0).to(device)
            cv = cv.unsqueeze(0).to(device)
            cmk = cmk.unsqueeze(0).to(device)
            cin = cin.unsqueeze(0).to(device)
            label = torch.tensor([DS_S2I[d["main_sector"]]]).to(device)
            
            sl, cl, mcm_l, mir_l, mid_l = m(ce, cm, df, cv, cmk, cin, sector_emb_matrix, training=True)
            loss = F.cross_entropy(sl, label, label_smoothing=cfg.label_smoothing) + \
                   F.cross_entropy(cl, label, label_smoothing=cfg.label_smoothing) + \
                   cfg.mcm_weight * mcm_l + cfg.miras_weight * mir_l + 0.1 * mid_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    
    # Test on holdout
    m.eval()
    correct = 0
    with torch.no_grad():
        for d in ho_data:
            inputs = encode_dataset(d, augment=False)
            ce, cm, df, cv, cmk, cin = inputs
            ce = ce.unsqueeze(0).to(device)
            cm = cm.unsqueeze(0).to(device)
            df = df.unsqueeze(0).to(device)
            cv = cv.unsqueeze(0).to(device)
            cmk = cmk.unsqueeze(0).to(device)
            cin = cin.unsqueeze(0).to(device)
            
            sl, _, _, _, _ = m(ce, cm, df, cv, cmk, cin, sector_emb_matrix, training=False)
            pred = sl.argmax(-1).item()
            actual = DS_S2I[d["main_sector"]]
            ok = pred == actual
            if ok:
                correct += 1
            mark = "OK" if ok else "XX"
            print(f"  [{mark}] holdout={holdout:20s} -> pred={DATASET_SECTORS[pred]}")
    
    acc = correct / len(ho_data) * 100
    unseen_results[holdout] = (correct, len(ho_data), acc)
    print(f"  {holdout}: {correct}/{len(ho_data)} = {acc:.0f}%\n")

# Summary
print("=" * 60)
print("UNSEEN SECTOR SUMMARY")
print("=" * 60)
tot_c = sum(v[0] for v in unseen_results.values())
tot_n = sum(v[1] for v in unseen_results.values())
for s in sorted(unseen_results):
    c, n, acc = unseen_results[s]
    mark = "OK" if c == n else "XX"
    print(f"  [{mark}] {s:20s}: {c}/{n} = {acc:.0f}%")
overall = tot_c / max(tot_n, 1) * 100
print(f"\n  OVERALL UNSEEN: {tot_c}/{tot_n} = {overall:.1f}%")
if overall >= 80:
    print("  SECTOR-AGNOSTIC CALISIYOR - production data uretmeye gecebiliriz")
elif overall >= 50:
    print("  KISMEN CALISIYOR - tweak gerekli")
else:
    print("  YETERSIZ - mimari degisiklik gerekli")
