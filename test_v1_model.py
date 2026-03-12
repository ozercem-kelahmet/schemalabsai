#!/usr/bin/env python3
"""
SchemaLabs V1 — Model Test & Evaluation
Loads checkpoint, runs full val evaluation with per-sector breakdown,
and interactive prediction mode.

Usage:
  python test_v1_model.py                    # full eval + interactive
  python test_v1_model.py --eval-only        # just evaluation
  python test_v1_model.py --predict-only     # just interactive prediction
"""
import json, os, math, sys, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG (must match training)
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
    midas_iterations = 10
    mcm_mask_ratio = 0.15
    miras_low_rank_k = 64

cfg = Config()
device = torch.device("cpu")  # test on CPU for stability

BASE = Path(os.path.expanduser("~/Desktop/schemalabsai")) if Path(os.path.expanduser("~/Desktop/schemalabsai")).exists() else Path("/opt/schemalabsai")
PRECOMP_DIR = BASE / "data" / "v1_precomputed"
CHECKPOINT_PATH = BASE / "checkpoints" / "schema_v1_production_best.pt"

if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = BASE / "checkpoints" / "schema_v1_production.pt"

print(f"Device: {device}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# ============================================================
# MODEL COMPONENTS (same as training)
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
        local_out = self.local_reason(cells, col_mask)
        B, R, C, d = local_out.shape
        miras_in = local_out.reshape(B, R*C, d)
        local_out = self.miras(miras_in).reshape(B, R, C, d)
        global_repr = self.global_reason(local_out, col_mask)
        schema_pool = (schema * col_mask.unsqueeze(-1).float()).sum(1) / col_mask.sum(1, keepdim=True).float().clamp(min=1)
        sector_logits = self.sector_head(global_repr, schema_pool, sector_emb_matrix)
        cls_logits = self.cls_head(global_repr)
        return sector_logits, cls_logits, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

# ============================================================
# LOAD DATA & MODEL
# ============================================================
print("\nLoading metadata...")
with open(PRECOMP_DIR / "metadata.json") as f:
    meta = json.load(f)
DS_SECTORS = meta["ds_sectors"]
DS_S2I = meta["ds_s2i"]
N_DS = len(DS_SECTORS)
I2S = {i: s for s, i in DS_S2I.items()}

sector_emb_matrix = torch.load(PRECOMP_DIR / "sector_emb_matrix.pt", weights_only=True).to(device)

print(f"Sectors ({N_DS}): {DS_SECTORS}")

print("\nLoading model...")
model = SchemaV1Production(cfg, N_DS).to(device)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

acc = ckpt.get("accuracy", ckpt.get("best_accuracy", "?"))
epoch = ckpt.get("epoch", "?")
print(f"Loaded: epoch={epoch}, accuracy={acc}%")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# ENCODE HELPERS (for interactive prediction)
# ============================================================
sbert = None  # lazy load

def get_sbert():
    global sbert
    if sbert is None:
        print("Loading SBERT...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return sbert

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

def encode_dataset(d):
    sb = get_sbert()
    columns = list(d["columns"][:cfg.max_cols])
    rows = d.get("sample_rows", [])[:cfg.max_rows]
    n_cols = len(columns)

    col_embs = torch.zeros(cfg.max_cols, cfg.sbert_dim)
    col_mask = torch.zeros(cfg.max_cols, dtype=torch.bool)
    for i, col in enumerate(columns[:cfg.max_cols]):
        col_embs[i] = torch.tensor(sb.encode([col.lower().replace("_", " ")], convert_to_numpy=True)[0])
        col_mask[i] = True

    dist_fps = torch.zeros(cfg.max_cols, cfg.fingerprint_dim)
    cell_values = torch.zeros(cfg.max_rows, cfg.max_cols)
    cell_mask = torch.zeros(cfg.max_rows, cfg.max_cols)
    cell_is_numeric = torch.zeros(cfg.max_rows, cfg.max_cols)

    for c_idx in range(min(n_cols, cfg.max_cols)):
        col_vals = [row[c_idx] if c_idx < len(row) else "" for row in rows]
        dist_fps[c_idx] = torch.tensor(compute_fingerprint(col_vals))
        for r_idx, v in enumerate(col_vals[:cfg.max_rows]):
            if v and str(v).strip():
                cell_mask[r_idx, c_idx] = 1
                if is_numeric(v):
                    val = parse_numeric(v)
                    cell_values[r_idx, c_idx] = math.copysign(math.log1p(abs(val)), val) / 20.0
                    cell_is_numeric[r_idx, c_idx] = 1

    return col_embs, col_mask, dist_fps, cell_values, cell_mask, cell_is_numeric

@torch.no_grad()
def predict(dataset_dict):
    ce, cm, df, cv, cmask, cin = encode_dataset(dataset_dict)
    ce, cm, df, cv, cmask, cin = [t.unsqueeze(0).to(device) for t in [ce, cm, df, cv, cmask, cin]]
    sl, cl, _, _, _ = model(ce, cm, df, cv, cmask, cin, sector_emb_matrix, training=False)

    # Top-5 sector predictions
    sector_probs = F.softmax(sl[0], dim=-1)
    top5_vals, top5_idx = sector_probs.topk(5)

    # Top-5 class predictions
    cls_probs = F.softmax(cl[0], dim=-1)
    ctop5_vals, ctop5_idx = cls_probs.topk(5)

    print("\n  SECTOR predictions:")
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_vals)):
        print(f"    {i+1}. {I2S[idx.item()]:30s} {prob.item()*100:.1f}%")

    print("  CLASS predictions:")
    for i, (idx, prob) in enumerate(zip(ctop5_idx, ctop5_vals)):
        print(f"    {i+1}. {I2S[idx.item()]:30s} {prob.item()*100:.1f}%")

    return I2S[top5_idx[0].item()], top5_vals[0].item()

# ============================================================
# FULL EVALUATION
# ============================================================
def run_full_eval():
    print("\n" + "=" * 60)
    print("FULL VALIDATION EVALUATION")
    print("=" * 60)

    # Load precomputed test data
    import random
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    col_embs_all = torch.load(PRECOMP_DIR / "col_embs.pt", weights_only=True)
    col_mask_all = torch.load(PRECOMP_DIR / "col_mask.pt", weights_only=True)
    dist_fps_all = torch.load(PRECOMP_DIR / "dist_fps.pt", weights_only=True)
    cell_values_all = torch.load(PRECOMP_DIR / "cell_values.pt", weights_only=True)
    cell_mask_all = torch.load(PRECOMP_DIR / "cell_mask.pt", weights_only=True)
    cell_is_num_all = torch.load(PRECOMP_DIR / "cell_is_numeric.pt", weights_only=True)
    labels_all = torch.load(PRECOMP_DIR / "labels.pt", weights_only=True)

    N = len(labels_all)
    all_indices = list(range(N))
    random.shuffle(all_indices)
    split = int(0.95 * N)
    test_indices = all_indices[split:]

    print(f"Test samples: {len(test_indices):,}")

    correct_s = 0
    correct_c = 0
    total = 0
    per_sector_correct = defaultdict(int)
    per_sector_total = defaultdict(int)
    confusion = defaultdict(Counter)  # true -> predicted counts

    t0 = time.time()
    for i, idx in enumerate(test_indices):
        ce = col_embs_all[idx].float().unsqueeze(0)
        cm = col_mask_all[idx].unsqueeze(0)
        df = dist_fps_all[idx].float().unsqueeze(0)
        cv = cell_values_all[idx].float().unsqueeze(0)
        cmask = cell_mask_all[idx].float().unsqueeze(0)
        cin = cell_is_num_all[idx].float().unsqueeze(0)
        lbl = labels_all[idx].item()

        sl, cl, _, _, _ = model(ce, cm.bool(), df, cv, cmask, cin, sector_emb_matrix, training=False)
        s_pred = sl.argmax(-1).item()
        c_pred = cl.argmax(-1).item()

        true_sector = I2S[lbl]
        pred_sector = I2S[s_pred]

        if s_pred == lbl:
            correct_s += 1
            per_sector_correct[true_sector] += 1
        if c_pred == lbl:
            correct_c += 1

        per_sector_total[true_sector] += 1
        confusion[true_sector][pred_sector] += 1
        total += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:,}/{len(test_indices):,}] s_acc={correct_s/total*100:.1f}% rate={rate:.0f}/s")

    elapsed = time.time() - t0
    s_acc = correct_s / total * 100
    c_acc = correct_c / total * 100

    print(f"\n{'='*60}")
    print(f"RESULTS ({total:,} test samples, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Sector Accuracy: {s_acc:.1f}%")
    print(f"  Class Accuracy:  {c_acc:.1f}%")

    print(f"\n{'='*60}")
    print(f"PER-SECTOR BREAKDOWN")
    print(f"{'='*60}")
    print(f"  {'Sector':30s} {'Correct':>8s} {'Total':>8s} {'Acc':>8s}")
    print(f"  {'-'*56}")
    for sector in sorted(per_sector_total.keys()):
        c = per_sector_correct.get(sector, 0)
        t = per_sector_total[sector]
        a = c / t * 100 if t > 0 else 0
        print(f"  {sector:30s} {c:8d} {t:8d} {a:7.1f}%")

    # Worst sectors
    print(f"\n{'='*60}")
    print(f"WORST 5 SECTORS (most confused)")
    print(f"{'='*60}")
    sector_acc = {s: per_sector_correct.get(s, 0) / per_sector_total[s] * 100 for s in per_sector_total}
    worst = sorted(sector_acc.items(), key=lambda x: x[1])[:5]
    for sector, acc in worst:
        print(f"\n  {sector} ({acc:.1f}% accuracy)")
        top_confused = confusion[sector].most_common(3)
        for pred, count in top_confused:
            pct = count / per_sector_total[sector] * 100
            marker = " ✓" if pred == sector else ""
            print(f"    → predicted as {pred:25s}: {count:5d} ({pct:.1f}%){marker}")

    return s_acc, c_acc

# ============================================================
# INTERACTIVE PREDICTION
# ============================================================
def run_interactive():
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION")
    print("=" * 60)

    # Example datasets for quick testing
    examples = [
        {
            "name": "Sales Data",
            "columns": ["product_name", "quantity", "unit_price", "total_revenue", "date", "region"],
            "sample_rows": [
                ["Widget A", "100", "9.99", "999.00", "2024-01-15", "North"],
                ["Widget B", "250", "4.99", "1247.50", "2024-01-15", "South"],
                ["Gadget C", "50", "24.99", "1249.50", "2024-01-16", "East"],
            ]
        },
        {
            "name": "Patient Records",
            "columns": ["patient_id", "age", "gender", "diagnosis", "blood_pressure", "heart_rate", "medication"],
            "sample_rows": [
                ["P001", "45", "M", "Hypertension", "140/90", "78", "Lisinopril"],
                ["P002", "62", "F", "Diabetes", "130/85", "82", "Metformin"],
                ["P003", "38", "M", "Asthma", "120/80", "72", "Albuterol"],
            ]
        },
        {
            "name": "Student Grades",
            "columns": ["student_name", "math_score", "science_score", "english_score", "gpa", "grade_level"],
            "sample_rows": [
                ["Alice", "92", "88", "95", "3.8", "10"],
                ["Bob", "78", "82", "71", "3.2", "10"],
                ["Charlie", "95", "97", "89", "3.9", "11"],
            ]
        },
        {
            "name": "Stock Market",
            "columns": ["ticker", "open_price", "close_price", "volume", "market_cap", "pe_ratio", "sector"],
            "sample_rows": [
                ["AAPL", "150.25", "152.30", "52000000", "2400000000000", "28.5", "Technology"],
                ["GOOGL", "2800.00", "2825.50", "1200000", "1800000000000", "25.2", "Technology"],
                ["JPM", "160.10", "161.75", "8500000", "470000000000", "12.1", "Financial"],
            ]
        },
        {
            "name": "Weather Data",
            "columns": ["city", "temperature_c", "humidity", "wind_speed_kmh", "precipitation_mm", "condition"],
            "sample_rows": [
                ["Istanbul", "22.5", "65", "15", "0", "Sunny"],
                ["London", "12.3", "82", "25", "5.2", "Rainy"],
                ["Tokyo", "28.1", "70", "10", "0", "Cloudy"],
            ]
        },
    ]

    print("\nTest datasets:")
    for i, ex in enumerate(examples):
        print(f"  {i+1}. {ex['name']} — columns: {ex['columns'][:5]}")

    print("\nRunning predictions...")
    for ex in examples:
        print(f"\n{'─'*50}")
        print(f"Dataset: {ex['name']}")
        print(f"Columns: {ex['columns']}")
        sector, conf = predict(ex)
        print(f"  → PREDICTED: {sector} ({conf*100:.1f}% confidence)")

    # Interactive loop
    print(f"\n{'='*60}")
    print("Enter your own dataset (or 'q' to quit)")
    print("Format: column1,column2,column3")
    print("Then enter rows: val1,val2,val3 (empty line to predict)")
    print(f"{'='*60}")

    while True:
        cols_input = input("\nColumns (comma-separated, or 'q'): ").strip()
        if cols_input.lower() == 'q':
            break

        columns = [c.strip() for c in cols_input.split(",")]
        rows = []
        print("Enter rows (empty line to predict):")
        while True:
            row_input = input("  > ").strip()
            if not row_input:
                break
            rows.append([v.strip() for v in row_input.split(",")])

        if columns:
            dataset = {"columns": columns, "sample_rows": rows}
            sector, conf = predict(dataset)
            print(f"\n  → PREDICTED: {sector} ({conf*100:.1f}% confidence)")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if "--predict-only" in sys.argv:
        run_interactive()
    elif "--eval-only" in sys.argv:
        run_full_eval()
    else:
        s_acc, c_acc = run_full_eval()
        run_interactive()
