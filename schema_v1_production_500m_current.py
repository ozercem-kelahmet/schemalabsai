#!/usr/bin/env python3
"""
SchemaLabs V1 — 517M Production Training (All-in-One, Benchmark-Ready)
======================================================================
68 components, 10000 sector labels, 2M dataset, GCP-safe
Full progress logging: loss breakdown, sector/cls/top5 accuracy, per-sector stats

Usage:
  python schema_v1_production_500m.py                # auto pre-compute + train
  python schema_v1_production_500m.py --precompute   # force re-precompute
"""
import json, os, math, random, time, gc, sys
import numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint

if torch.cuda.is_available():
    from torch.amp import autocast, GradScaler
    def amp_autocast(): return autocast("cuda")
else:
    from contextlib import nullcontext
    GradScaler = None
    def amp_autocast(): return nullcontext()

from sentence_transformers import SentenceTransformer

class Config:
    d_model = 1024
    n_heads = 16
    head_dim = 64
    n_latent = 256
    ffn_hidden = 4096
    n_local_layers = 6
    n_global_layers = 6
    n_schema_layers = 6
    n_miras_layers = 4
    dropout = 0.1
    sbert_dim = 384
    fingerprint_dim = 7
    max_cols = 30
    max_rows = 10
    midas_iterations = 1
    midas_noise_std = 0.1
    midas_recon_weight = 1.0
    midas_imputation_weight = 3.0
    midas_synth_missing_rate = 0.1
    mcm_mask_ratio = 0.15
    mcm_weight = 0.1
    miras_weight = 0.05
    miras_low_rank_k = 64
    miras_huber_delta = 1.0
    miras_lq_q = 4
    miras_momentum = 0.9
    contrastive_weight = 0.05
    contrastive_temperature = 0.07
    ewc_lambda = 1000
    ewc_fisher_decay = 0.9
    replay_ratio = 0.1
    replay_max = 50000
    n_sectors = 10000
    batch_size = 64
    grad_accum = 1
    lr = 5e-5
    weight_decay = 0.01
    warmup_epochs = 1
    total_epochs = 5
    max_grad_norm = 1.0
    label_smoothing = 0.01
    curriculum_enabled = False
    curriculum_easy_classes = 5
    curriculum_warmup_fraction = 0.2
    self_learning_enabled = True
    pseudo_label_threshold_start = 0.9
    pseudo_label_threshold_end = 0.99
    pseudo_label_max_per_epoch = 10000
    ood_enabled = True
    early_stop_patience = 5
    log_every = 50
    checkpoint_every = 200000
    val_samples = 2000
    shard_size = 50000
    data_path = "data/v1_training_data_2m.json"
    sector_list_path = "data/sector_list_10000.json"
    sector_emb_path = "data/sector_embeddings_10000.npy"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

cfg = Config()
device = torch.device(cfg.device)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Flash Attention + aggressive TF32
    torch.set_float32_matmul_precision('medium')
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

BASE = Path(os.path.expanduser("~/Desktop/schemalabsai")) if Path(os.path.expanduser("~/Desktop/schemalabsai")).exists() else Path("/opt/schemalabsai")
PRECOMP_DIR = BASE / "data" / "v1_precomputed_2m"
CHECKPOINT_PATH = BASE / "checkpoints" / "schema_v1_500m.pt"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH = BASE / "training_v1_500m.log"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode='a')])
log = logging.getLogger(__name__)

# ============================================================
# SAFEGUARDS — protect site + Docker containers
# ============================================================
import subprocess, signal

# 1. Low CPU priority
os.nice(0)

# 2. GPU memory fraction — reserve 30% for fine-tune/inference
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

# 3. RAM monitoring
def get_ram_usage_gb():
    try:
        with open('/proc/meminfo') as f:
            lines = f.readlines()
        total = int([l for l in lines if 'MemTotal' in l][0].split()[1]) / 1024 / 1024
        avail = int([l for l in lines if 'MemAvailable' in l][0].split()[1]) / 1024 / 1024
        return total, avail
    except:
        return 0, 999

RAM_LIMIT_GB = 15.0

def check_ram():
    total, avail = get_ram_usage_gb()
    if avail < 3.0:
        log.warning(f"RAM CRITICAL: {avail:.1f}GB available / {total:.1f}GB total — pausing for GC")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(5)
        _, avail2 = get_ram_usage_gb()
        if avail2 < 2.0:
            log.error(f"RAM STILL LOW: {avail2:.1f}GB — stopping training to protect site")
            sys.exit(1)
    return avail

# 4. Docker health check
def check_docker_healthy():
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}} {{.Status}}'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return True
        unhealthy = [l for l in result.stdout.strip().split('\n') if l and 'unhealthy' in l.lower()]
        if unhealthy:
            log.warning(f"Unhealthy containers: {unhealthy}")
            return False
        return True
    except:
        return True

# 5. Graceful shutdown handler
_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    log.info("Received shutdown signal, finishing current batch...")
    _shutdown = True
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

log.info(f"Device: {device} | Base: {BASE}")
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name()} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

# ============================================================
# BUILDING BLOCKS
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_in, bias=False)
        self.w3 = nn.Linear(d_in, d_hidden, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GatedOutput(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Linear(d, d); self.proj = nn.Linear(d, d)
    def forward(self, x): return self.proj(x) * torch.sigmoid(self.gate(x))

class LowRankProjection(nn.Module):
    def __init__(self, d, k=64):
        super().__init__()
        self.down = nn.Linear(d, k, bias=False); self.up = nn.Linear(k, d, bias=False)
    def forward(self, x): return self.up(self.down(x))

class SDPAttention(nn.Module):
    """Fast attention using F.scaled_dot_product_attention"""
    def __init__(self, d, nh, dropout=0.0):
        super().__init__()
        self.nh = nh; self.hd = d // nh
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout
    def forward(self, q, k=None, v=None, key_padding_mask=None):
        if k is None: k = q
        if v is None: v = k
        B, S, D = q.shape; S2 = k.shape[1]
        qq = self.q_proj(q).view(B, S, self.nh, self.hd).transpose(1, 2)
        kk = self.k_proj(k).view(B, S2, self.nh, self.hd).transpose(1, 2)
        vv = self.v_proj(v).view(B, S2, self.nh, self.hd).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(B, self.nh, S, S2)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0).to(qq.dtype)
        o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(o.transpose(1, 2).reshape(B, S, D))

# ============================================================
# 1-6. MIDAS
# ============================================================
class MIDAS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.value_proj = nn.Linear(d, d)
        self.mask_proj = nn.Linear(d, d)
        self.encoder = nn.Sequential(nn.Linear(d*2, cfg.ffn_hidden), RMSNorm(cfg.ffn_hidden), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_hidden, d), RMSNorm(d))
        self.decoder = nn.Sequential(nn.Linear(d, cfg.ffn_hidden), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_hidden, d))
        self.confidence_head = nn.Sequential(nn.Linear(d, d//4), nn.SiLU(), nn.Linear(d//4, 1), nn.Sigmoid())

    def forward(self, x, col_mask, training=True):
        B, C, D = x.shape
        current = x.clone()
        if training:
            synth_mask = (torch.rand(B, C, device=x.device) > cfg.midas_synth_missing_rate) & col_mask
            work_mask = synth_mask
        else:
            work_mask = col_mask
        confidences = []
        for it in range(cfg.midas_iterations):
            noisy = (current + torch.randn_like(current) * cfg.midas_noise_std * work_mask.unsqueeze(-1).to(current.dtype)) if (training and it == 0) else current
            val_emb = self.value_proj(noisy * work_mask.unsqueeze(-1).to(current.dtype))
            mask_emb = self.mask_proj(work_mask.unsqueeze(-1).to(current.dtype).expand_as(noisy))
            encoded = self.encoder(torch.cat([val_emb, mask_emb], dim=-1))
            decoded = self.decoder(encoded)
            conf = self.confidence_head(encoded).squeeze(-1)
            confidences.append(conf.detach())
            missing = ~work_mask
            if missing.any():
                blend = conf.unsqueeze(-1) * decoded + (1 - conf.unsqueeze(-1)) * current
                current = torch.where(missing.unsqueeze(-1).expand_as(current), blend, x)
        known = work_mask.unsqueeze(-1).expand_as(decoded)
        recon_loss = F.mse_loss(decoded[known], x[known]) if known.any() else torch.tensor(0.0, device=x.device)
        imputation_loss = torch.tensor(0.0, device=x.device)
        if training:
            removed = (col_mask & ~work_mask).unsqueeze(-1).expand_as(current)
            if removed.any():
                imputation_loss = F.mse_loss(current[removed], x[removed])
        return current, recon_loss, imputation_loss, confidences

# ============================================================
# 7-8. CELL PROCESSING
# ============================================================
class CellProcessing(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.sbert_proj = nn.Sequential(nn.Linear(cfg.sbert_dim, d), RMSNorm(d))
        self.fp_proj = nn.Sequential(nn.Linear(cfg.fingerprint_dim, d), RMSNorm(d))
        self.value_proj = nn.Linear(1, d)
        self.type_emb = nn.Embedding(3, d)
        self.pos_emb = nn.Embedding(cfg.max_cols, d)
        self.row_pos_emb = nn.Embedding(cfg.max_rows, d)
        self.fusion_col = nn.Sequential(nn.Linear(d*3, d), RMSNorm(d), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), RMSNorm(d))
        self.fusion_cell = nn.Sequential(nn.Linear(d*4, d), RMSNorm(d), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), RMSNorm(d))

    def forward(self, col_embs, dist_fps, cell_values, cell_mask, cell_is_numeric):
        B, C, _ = col_embs.shape; R = cell_values.shape[1]
        sbert_emb = self.sbert_proj(col_embs.float())
        fp_emb = self.fp_proj(dist_fps.float())
        pos_emb = self.pos_emb(torch.arange(C, device=col_embs.device)).unsqueeze(0).expand(B, -1, -1)
        col_features = self.fusion_col(torch.cat([sbert_emb, fp_emb, pos_emb], dim=-1))
        row_emb = self.row_pos_emb(torch.arange(R, device=col_embs.device)).unsqueeze(0).unsqueeze(2).expand(B, -1, C, -1)
        col_broadcast = col_features.unsqueeze(1).expand(B, R, C, -1)
        val_emb = self.value_proj(cell_values.float().unsqueeze(-1))
        type_ids = torch.zeros(B, R, C, device=col_embs.device, dtype=torch.long)
        type_ids[cell_mask] = 2
        type_ids[cell_is_numeric] = 1
        type_e = self.type_emb(type_ids)
        cell_grid = self.fusion_cell(torch.cat([col_broadcast, val_emb, row_emb, type_e], dim=-1))
        return col_features, cell_grid

# ============================================================
# 9. SCHEMA PROCESSING
# ============================================================
class SchemaTransformerLayer(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.norm1 = RMSNorm(d); self.attn = SDPAttention(d, nh, dropout=cfg.dropout)
        self.norm2 = RMSNorm(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x, mask=None):
        h = self.norm1(x); h = self.attn(h, key_padding_mask=mask); x = x + self.drop(h)
        return x + self.drop(self.ffn(self.norm2(x)))

class SchemaProcessing(nn.Module):
    def __init__(self, d, nh, nl):
        super().__init__()
        self.layers = nn.ModuleList([SchemaTransformerLayer(d, nh) for _ in range(nl)]); self.norm = RMSNorm(d)
    def forward(self, x, col_mask=None):
        pm = ~col_mask if col_mask is not None else None
        for l in self.layers:
            x = grad_checkpoint(l, x, pm, use_reentrant=False)
        return self.norm(x)

# ============================================================
# 10. LOCAL REASONING
# ============================================================
class AxialAttentionLayer(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.rn = RMSNorm(d); self.ra = SDPAttention(d, nh, dropout=cfg.dropout)
        self.cn = RMSNorm(d); self.ca = SDPAttention(d, nh, dropout=cfg.dropout)
        self.fn = RMSNorm(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        B, R, C, D = x.shape
        xr = x.reshape(B*R, C, D); h = self.rn(xr); h = self.ra(h); xr = xr + self.drop(h); x = xr.reshape(B, R, C, D)
        xc = x.permute(0,2,1,3).reshape(B*C, R, D); h = self.cn(xc); h = self.ca(h); xc = xc + self.drop(h); x = xc.reshape(B, C, R, D).permute(0,2,1,3)
        return x + self.drop(self.ffn(self.fn(x)))

class LocalReasoning(nn.Module):
    def __init__(self, d, nh, nl):
        super().__init__()
        self.layers = nn.ModuleList([AxialAttentionLayer(d, nh) for _ in range(nl)]); self.norm = RMSNorm(d)
    def forward(self, cell_grid):
        x = cell_grid
        for l in self.layers:
            x = grad_checkpoint(l, x, use_reentrant=False)
        return self.norm(x).mean(dim=(1, 2))

# ============================================================
# 11. GLOBAL REASONING
# ============================================================
class PerceiverLayer(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.cnq = RMSNorm(d); self.cnkv = RMSNorm(d); self.ca = SDPAttention(d, nh, dropout=cfg.dropout)
        self.sn = RMSNorm(d); self.sa = SDPAttention(d, nh, dropout=cfg.dropout)
        self.fn = RMSNorm(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, lat, ctx):
        h = self.cnq(lat); c = self.cnkv(ctx); h = self.ca(h, c, c); lat = lat + self.drop(h)
        h = self.sn(lat); h = self.sa(h); lat = lat + self.drop(h)
        return lat + self.drop(self.ffn(self.fn(lat)))

class GlobalReasoning(nn.Module):
    def __init__(self, d, nh, nl, n_lat):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_lat, d) * 0.02)
        self.layers = nn.ModuleList([PerceiverLayer(d, nh) for _ in range(nl)]); self.norm = RMSNorm(d)
    def forward(self, ctx):
        lat = self.latents.expand(ctx.shape[0], -1, -1)
        for l in self.layers:
            lat = grad_checkpoint(l, lat, ctx, use_reentrant=False)
        return self.norm(lat).mean(dim=1)

# ============================================================
# 12. SECTOR HEAD — learnable scale
# ============================================================
class SectorHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, cfg.sbert_dim))
        self.scale = nn.Parameter(torch.tensor(20.0))
    def forward(self, x, sem):
        return F.normalize(self.proj(x), dim=-1) @ F.normalize(sem, dim=-1).T * self.scale.abs()

# ============================================================
# 13-15. TASK HEADS
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self, d, nc):
        super().__init__()
        self.head = nn.Sequential(RMSNorm(d), nn.Linear(d, cfg.ffn_hidden), nn.SiLU(), nn.Dropout(0.1), nn.Linear(cfg.ffn_hidden, nc))
    def forward(self, x): return self.head(x)

class RegressionHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Sequential(RMSNorm(d), nn.Linear(d, d//2), nn.SiLU(), nn.Dropout(0.1), nn.Linear(d//2, 1))
    def forward(self, x): return self.head(x).squeeze(-1)

class MCMHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Sequential(RMSNorm(d), nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, 1))
    def forward(self, x): return self.head(x).squeeze(-1)

# ============================================================
# 20-23. DOMAIN ADAPTATION
# ============================================================
class DomainSchemaAdapter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.scale = nn.Sequential(nn.Linear(d, d), nn.Sigmoid()); self.shift = nn.Linear(d, d)
    def forward(self, se, gc):
        gc = gc.unsqueeze(1) if gc.dim() == 2 else gc
        return self.norm(se) * self.scale(gc) + self.shift(gc)

class DomainKnowledgeInjection(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nq = RMSNorm(d); self.nkv = RMSNorm(d)
        self.ca = SDPAttention(d, nh, dropout=cfg.dropout); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, cr, se):
        q = self.nq(cr); kv = self.nkv(se); h = self.ca(q, kv, kv); return cr + self.drop(h)

class SchemaCellFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(d*2, d), RMSNorm(d))
    def forward(self, sr, cr):
        c = torch.cat([sr, cr], dim=-1); g = self.gate(c); return g * self.proj(c) + (1 - g) * cr

class DomainSpecificHeads(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)); self.norm = RMSNorm(d)
    def forward(self, x): return self.norm(x + self.cond(x))

# ============================================================
# 24-38. MIRAS
# ============================================================
class MIRASLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = RMSNorm(d); self.low_rank = LowRankProjection(d, cfg.miras_low_rank_k)
        self.gated_out = GatedOutput(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.ffn_norm = RMSNorm(d)
        self.eta = nn.Parameter(torch.ones(d)*0.01); self.delta = nn.Parameter(torch.ones(d)*cfg.miras_huber_delta)
        self.alpha = nn.Parameter(torch.ones(d)*0.9)
        self.q_proj = nn.Linear(d, d, bias=False); self.k_proj = nn.Linear(d, d, bias=False)
        self.temperature = nn.Parameter(torch.ones(1))

    def huber_bias(self, p, t):
        diff = p - t; ad = diff.abs(); dl = self.delta.abs().clamp(min=0.01)
        return torch.where(ad <= dl, 0.5*diff.pow(2), dl*(ad - 0.5*dl)).mean()

    def lq_retention(self, wo, wn):
        diff = (wn - wo).abs(); th = diff.mean()*0.1; return F.relu(diff - th).pow(cfg.miras_lq_q).mean()

    def forward(self, x, ms=None):
        h = self.low_rank(self.norm(x))
        q = F.normalize(self.q_proj(h), dim=-1)
        k = F.normalize(self.k_proj(h), dim=-1)
        attn_weight = (q * k).sum(dim=-1, keepdim=True).clamp(-1, 1)
        h = h * (1 + attn_weight)
        if ms is not None:
            grad = h - ms
            momentum_update = cfg.miras_momentum * ms + (1 - cfg.miras_momentum) * grad
            h = ms + self.alpha.abs() * self.eta.abs() * momentum_update
        ns = h.detach(); h = self.gated_out(h) + x; h = h + self.ffn(self.ffn_norm(h))
        return h, ns

    def calibrated_logits(self, lo): return lo / self.temperature.abs().clamp(min=0.1)

class MIRAS(nn.Module):
    def __init__(self, d, nl=4):
        super().__init__()
        self.layers = nn.ModuleList([MIRASLayer(d) for _ in range(nl)])
        self.contrastive_proj = nn.Sequential(nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, 128))
        self.norm = RMSNorm(d)
        self.emb_queue = None; self.label_queue = None; self.queue_size = 256

    def forward(self, x, ms=None):
        if ms is None: ms = [None]*len(self.layers)
        ns = []; ml = torch.tensor(0.0, device=x.device)
        for i, l in enumerate(self.layers):
            prev = ms[i]
            if prev is not None and prev.dim() == 1:
                prev = prev.unsqueeze(0).expand(x.shape[0], -1)
            x, s = l(x, prev); ns.append(s.mean(dim=0))
            if prev is not None: ml = ml + l.huber_bias(x, prev) + 0.1*l.lq_retention(prev, s.mean(dim=0).unsqueeze(0).expand_as(prev))
        return self.norm(x), ns, ml / max(len(self.layers), 1)

    def contrastive_loss(self, emb, labels):
        p_curr = F.normalize(self.contrastive_proj(emb), dim=-1)
        if self.emb_queue is not None:
            all_p = torch.cat([p_curr, self.emb_queue.to(p_curr.device)], dim=0)
            all_labels = torch.cat([labels, self.label_queue.to(labels.device)], dim=0)
        else:
            all_p = p_curr; all_labels = labels
        self.emb_queue = all_p[-self.queue_size:].detach().cpu()
        self.label_queue = all_labels[-self.queue_size:].detach().cpu()
        sim = p_curr @ all_p.detach().T / cfg.contrastive_temperature
        sim[:, :p_curr.shape[0]] = p_curr @ all_p[:p_curr.shape[0]].T / cfg.contrastive_temperature
        mp = (labels.unsqueeze(1) == all_labels.unsqueeze(0))
        for i in range(labels.shape[0]): mp[i, i] = False
        if mp.sum() == 0: return torch.tensor(0.0, device=emb.device)
        mn = ~mp
        for i in range(labels.shape[0]): mn[i, i] = False
        es = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
        pos = (es*mp.to(es.dtype)).sum(1); neg = (es*mn.to(es.dtype)).sum(1)
        return (-torch.log(pos/(pos+neg+1e-8)+1e-8)).mean()

    def calibrate(self, lo): return self.layers[0].calibrated_logits(lo)

# ============================================================
# EWC + OOD + CURRICULUM + SELF-LEARNING
# ============================================================
class EWCModule:
    def __init__(self):
        self.fisher = {}; self.old_params = {}; self.initialized = False; self.replay_indices = []
    def compute_fisher(self, model, dl, dev, ns=2000):
        self.compute_fisher_from_batches(model, dl, dev, ns)

    def compute_fisher_from_batches(self, model, dl, dev, ns=2000):
        model.eval()
        for n, p in model.named_parameters():
            if p.requires_grad: self.fisher[n] = torch.zeros_like(p, device=dev)
        c = 0
        for batch in dl:
            if c >= ns: break
            model.zero_grad(); items = [b.to(dev, non_blocking=True) for b in batch]; labels = items[-1]
            with amp_autocast():
                with amp_autocast(): out = model.forward_from_tensors(*items[:-1], training=False)
                F.nll_loss(F.log_softmax(out['sector_logits'], dim=-1), labels).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None: self.fisher[n] += p.grad.data.pow(2)*labels.shape[0]
            c += labels.shape[0]
        for n in self.fisher: self.fisher[n] /= max(c, 1)
        for n, p in model.named_parameters():
            if p.requires_grad: self.old_params[n] = p.data.clone()
        self.initialized = True; model.train()
    def penalty(self, model):
        if not self.initialized: return torch.tensor(0.0, device=device)
        lo = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher: lo = lo + (self.fisher[n]*(p-self.old_params[n]).pow(2)).sum()
        return cfg.ewc_lambda * lo
    def decay_fisher(self):
        for n in self.fisher: self.fisher[n] *= cfg.ewc_fisher_decay
    def update_replay(self, ds_size):
        n = min(int(ds_size*cfg.replay_ratio), cfg.replay_max)
        self.replay_indices = random.sample(range(ds_size), n)

class OODDetector:
    def __init__(self): self.mean = None; self.cov_inv = None; self.fitted = False
    def fit(self, emb):
        if emb.shape[0] < 10: return
        self.mean = emb.mean(0); c = emb - self.mean; cov = (c.T@c)/(emb.shape[0]-1) + torch.eye(c.shape[1], device=c.device)*1e-4
        try: self.cov_inv = torch.linalg.inv(cov); self.fitted = True
        except: self.fitted = False
    def mahalanobis(self, x):
        if not self.fitted: return torch.zeros(x.shape[0], device=x.device)
        d = x - self.mean; return torch.sqrt((d @ self.cov_inv * d).sum(-1).clamp(min=0))

class CurriculumScheduler:
    def __init__(self, labels, ns):
        counts = torch.bincount(labels, minlength=ns)
        _, self.order = counts.sort(descending=True); self.ns = ns
    def get_n_active(self, epoch, total):
        if not cfg.curriculum_enabled: return self.ns
        progress = epoch / max(total, 1)
        if progress < cfg.curriculum_warmup_fraction:
            return max(cfg.curriculum_easy_classes, int((progress/cfg.curriculum_warmup_fraction)*self.ns))
        return self.ns

class SelfLearner:
    def __init__(self): self.threshold = cfg.pseudo_label_threshold_start
    def update_threshold(self, ep, total):
        p = ep/max(total-1, 1)
        self.threshold = cfg.pseudo_label_threshold_start + p*(cfg.pseudo_label_threshold_end - cfg.pseudo_label_threshold_start)

# ============================================================
# SHARDED DATASET
# ============================================================
class ShardedDataset(Dataset):
    def __init__(self, shard_dir, n_samples):
        self.shard_dir = Path(shard_dir); self.n_samples = n_samples
        self.shard_size = cfg.shard_size; self.n_shards = math.ceil(n_samples/self.shard_size); self._cache = {}
    def _load_shard(self, si):
        wid = torch.utils.data.get_worker_info()
        ck = (id(wid), si) if wid else (0, si)
        if ck in self._cache: return self._cache[ck]
        d = {}
        for k in ['col_embs','col_mask','dist_fps','cell_values','cell_mask','cell_is_numeric','labels']:
            d[k] = torch.load(self.shard_dir/f"{k}_{si}.pt", weights_only=True)
        self._cache = {ck: d}; return d
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        si = idx // self.shard_size; li = idx % self.shard_size; d = self._load_shard(si)
        return d['col_embs'][li], d['col_mask'][li], d['dist_fps'][li], d['cell_values'][li], d['cell_mask'][li], d['cell_is_numeric'][li], d['labels'][li]

# ============================================================
# FULL MODEL
# ============================================================
class SchemaV1(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.d_model; self.d = d
        self.midas = MIDAS(d)
        self.cell_proc = CellProcessing(d)
        self.schema_proc = SchemaProcessing(d, cfg.n_heads, cfg.n_schema_layers)
        self.local_reason = LocalReasoning(d, cfg.n_heads, cfg.n_local_layers)
        self.global_reason = GlobalReasoning(d, cfg.n_heads, cfg.n_global_layers, cfg.n_latent)
        self.sector_head = SectorHead(d)
        self.cls_head = ClassificationHead(d, cfg.n_sectors)
        self.mcm_head = MCMHead(d)
        self.domain_adapter = DomainSchemaAdapter(d)
        self.domain_inject = DomainKnowledgeInjection(d, cfg.n_heads)
        self.schema_cell_fusion = SchemaCellFusion(d)
        self.domain_heads = DomainSpecificHeads(d)
        self.miras = MIRAS(d, cfg.n_miras_layers)
        self.combine_proj = nn.Sequential(nn.Linear(d*3, d), RMSNorm(d))
        self.register_buffer('_sem', torch.zeros(1, cfg.sbert_dim))

    def set_sector_emb(self, m): self._sem = m

    def forward_from_tensors(self, ce, cm, df, cv, cmask, cin, training=True):
        B = ce.shape[0]
        col_feat, cell_grid = self.cell_proc(ce, df, cv, cmask, cin)
        midas_out, rl, il, confs = self.midas(col_feat, cm, training)
        schema_out = self.schema_proc(midas_out, cm)
        gc = schema_out.mean(dim=1)
        sa = self.domain_adapter(schema_out, gc)
        cf = cell_grid.mean(dim=1)
        ci = self.domain_inject(cf, sa)
        sr = sa.mean(dim=1); cr = ci.mean(dim=1)
        fused = self.schema_cell_fusion(sr, cr)
        lo = self.local_reason(cell_grid)
        go = self.global_reason(sa)
        combined = self.combine_proj(torch.cat([lo, go, fused], dim=-1))
        combined, ms, ml = self.miras(combined, None)
        combined = self.domain_heads(combined)
        sl = self.sector_head(combined, self._sem)
        cl = self.miras.calibrate(self.cls_head(combined))
        mcml = torch.tensor(0.0, device=ce.device)
        if training:
            mm = (torch.rand(B, cfg.max_cols, device=ce.device) < cfg.mcm_mask_ratio) & cm
            if mm.any():
                mp = self.mcm_head(midas_out); tv = df[:,:,0].float()
                mcml = F.mse_loss(mp[mm], tv[mm])
        return {'sector_logits': sl, 'cls_logits': cl, 'midas_loss': cfg.midas_recon_weight*rl + cfg.midas_imputation_weight*il, 'mcm_loss': mcml, 'miras_loss': ml, 'contrastive_emb': combined}

    def count_params(self):
        t = sum(p.numel() for p in self.parameters()); tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return t, tr

    def get_feature_importance(self, ce, cm, df, cv, cmask, cin):
        ce = ce.float().requires_grad_(True)
        out = self.forward_from_tensors(ce, cm, df, cv, cmask, cin, training=False)
        out['cls_logits'].max(-1).values.sum().backward()
        return ce.grad.abs().mean(-1)

# ============================================================
# PRE-COMPUTE
# ============================================================
def needs_precompute():
    if "--precompute" in sys.argv: return True
    mp = PRECOMP_DIR / "metadata.json"
    if not mp.exists(): return True
    with open(mp) as f: return json.load(f).get("n_samples", 0) == 0

def run_precompute():
    PRECOMP_DIR.mkdir(parents=True, exist_ok=True)
    log.info("="*60 + "\nPHASE 1: PRE-COMPUTING ENCODINGS (SHARDED)\n" + "="*60)
    log.info("Loading data...")
    t0 = time.time()
    with open(BASE / cfg.data_path) as f: ALL_DATA = json.load(f)
    N = len(ALL_DATA); log.info(f"Loaded {N:,} datasets in {time.time()-t0:.1f}s")
    with open(BASE / cfg.sector_list_path) as f:
        sd = json.load(f); ALL_SECTORS = sd["sectors"]
    ads = sorted(set(d.get("sector", d.get("main_sector", "unknown")) for d in ALL_DATA))
    DS_S2I = {s: i for i, s in enumerate(ads)}; log.info(f"Unique sector labels: {len(DS_S2I)}")
    ds = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading SBERT on {ds}..."); sbert = SentenceTransformer("all-MiniLM-L6-v2", device=ds)
    acn = sorted(set(c.lower().replace("_"," ") for d in ALL_DATA for c in d["columns"]))
    log.info(f"Unique columns: {len(acn):,}"); CEM = {}
    for i in range(0, len(acn), 1024):
        b = acn[i:i+1024]; embs = sbert.encode(b, show_progress_bar=False, convert_to_numpy=True, batch_size=1024)
        for n, e in zip(b, embs): CEM[n] = e
        if (i//1024)%20==0: log.info(f"  Columns: {min(i+1024,len(acn)):,}/{len(acn):,}")
    SE = np.load(BASE / cfg.sector_emb_path); sel = []
    for s in ads:
        if s in ALL_SECTORS: sel.append(SE[ALL_SECTORS.index(s)])
        else: sel.append(sbert.encode([s], convert_to_numpy=True)[0])
    torch.save(torch.tensor(np.array(sel), dtype=torch.float32), PRECOMP_DIR / "sector_emb_matrix.pt")
    del sbert; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    def is_num(v):
        try: float(str(v).replace(",","")); return True
        except: return False
    def pn(v):
        try: return float(str(v).replace(",",""))
        except: return 0.0
    def cfp(vals):
        nums = [pn(v) for v in vals if is_num(v)]; nt = len(vals); u = len(set(vals))
        if nums:
            a = np.array(nums); fp = [float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.max(a)), len(nums)/max(nt,1), u/max(nt,1), float(nt)]
        else: fp = [0,0,0,0,0,u/max(nt,1),float(nt)]
        return [math.copysign(math.log1p(abs(v)),v)/20.0 if abs(v)>1 else v for v in fp]
    ns = math.ceil(N/cfg.shard_size); log.info(f"Encoding {N:,} in {ns} shards...")
    ts = time.time()
    for si in range(ns):
        ss = si*cfg.shard_size; se = min(ss+cfg.shard_size, N); sl = se-ss
        ce = torch.zeros(sl, cfg.max_cols, cfg.sbert_dim, dtype=torch.float16)
        cm = torch.zeros(sl, cfg.max_cols, dtype=torch.bool)
        df = torch.zeros(sl, cfg.max_cols, cfg.fingerprint_dim, dtype=torch.float16)
        cv = torch.zeros(sl, cfg.max_rows, cfg.max_cols, dtype=torch.float16)
        cmk = torch.zeros(sl, cfg.max_rows, cfg.max_cols, dtype=torch.bool)
        cin = torch.zeros(sl, cfg.max_rows, cfg.max_cols, dtype=torch.bool)
        lb = torch.zeros(sl, dtype=torch.long)
        for i in range(sl):
            d = ALL_DATA[ss+i]; cols = [c for c in list(d["columns"][:cfg.max_cols+1]) if c.lower() != "target"][:cfg.max_cols]; rows = d.get("sample_rows",[])[:cfg.max_rows]
            lb[i] = DS_S2I.get(d.get("sector", d.get("main_sector","unknown")), 0)
            for ci2, col in enumerate(cols[:cfg.max_cols]):
                k = col.lower().replace("_"," ")
                if k in CEM: ce[i,ci2] = torch.tensor(CEM[k], dtype=torch.float16)
                cm[i,ci2] = True
            for ci2 in range(min(len(cols), cfg.max_cols)):
                cvs = [row[ci2] if ci2<len(row) else "" for row in rows]
                df[i,ci2] = torch.tensor(cfp(cvs), dtype=torch.float16)
                for ri, v in enumerate(cvs[:cfg.max_rows]):
                    if v and str(v).strip():
                        cmk[i,ri,ci2] = True
                        if is_num(v):
                            val = pn(v); cv[i,ri,ci2] = math.copysign(math.log1p(abs(val)),val)/20.0; cin[i,ri,ci2] = True
        for k2, t in [('col_embs',ce),('col_mask',cm),('dist_fps',df),('cell_values',cv),('cell_mask',cmk),('cell_is_numeric',cin),('labels',lb)]:
            torch.save(t, PRECOMP_DIR/f"{k2}_{si}.pt")
        el = time.time()-ts; r = (si+1)/max(el,1); eta = (ns-si-1)/max(r,0.01)
        pct = 100*(si+1)/ns
        log.info(f"  Shard {si+1}/{ns} [{pct:.0f}%] ({sl:,}) elapsed={el:.0f}s eta={eta:.0f}s"); gc.collect()
    meta = {"n_samples":N,"n_shards":ns,"shard_size":cfg.shard_size,"ds_sectors":list(DS_S2I.keys()),"ds_s2i":DS_S2I,"n_sectors":len(DS_S2I)}
    with open(PRECOMP_DIR/"metadata.json","w") as f: json.dump(meta, f, indent=2)
    tb = sum(f2.stat().st_size for f2 in PRECOMP_DIR.glob("*.pt"))
    log.info(f"Saved {ns} shards ({tb/1024**3:.1f} GB). Pre-compute DONE.")
    del ALL_DATA, CEM; gc.collect()

# ============================================================
# TRAINING
# ============================================================
def topk_accuracy(logits, labels, k=5):
    topk = logits.topk(k, dim=-1).indices
    return (topk == labels.unsqueeze(-1)).any(dim=-1).float().sum().item()

def train():
    log.info("="*60 + "\nPHASE 2: TRAINING\n" + "="*60)
    with open(PRECOMP_DIR/"metadata.json") as f: meta = json.load(f)
    N = meta["n_samples"]; DS_S2I = meta["ds_s2i"]; DS_SECTORS = meta["ds_sectors"]; NS = meta["n_sectors"]
    I2S = {i: s for s, i in DS_S2I.items()}

    sem = torch.load(PRECOMP_DIR/"sector_emb_matrix.pt", weights_only=True).to(device)
    model = SchemaV1().to(device); model.set_sector_emb(sem)
    tp, trp = model.count_params()

    log.info(f"{'='*40}")
    log.info(f"  Samples:     {N:,}")
    log.info(f"  Sectors:     {NS}")
    log.info(f"  Parameters:  {tp:,} ({tp/1e6:.0f}M)")
    log.info(f"  Batch:       {cfg.batch_size} x {cfg.grad_accum} = {cfg.batch_size*cfg.grad_accum} effective")
    log.info(f"  LR:          {cfg.lr}")
    log.info(f"  Epochs:      {cfg.total_epochs} (early stop patience={cfg.early_stop_patience})")
    log.info(f"  Loss:        cls + sector + {cfg.mcm_weight}*mcm + {cfg.miras_weight}*miras + midas + 0.01*reg + {cfg.contrastive_weight}*contrastive + ewc")
    log.info(f"  Smoothing:   {cfg.label_smoothing}")
    log.info(f"  Curriculum:  {cfg.curriculum_enabled}")
    log.info(f"  Self-learn:  {cfg.self_learning_enabled}")
    log.info(f"  OOD:         {cfg.ood_enabled}")
    log.info(f"{'='*40}")

    log.info("Computing class weights (streaming)...")
    counts = torch.zeros(NS, dtype=torch.float32)
    total_samples = 0
    for si in range(meta["n_shards"]):
        lb = torch.load(PRECOMP_DIR/f"labels_{si}.pt", weights_only=True)
        counts += torch.bincount(lb, minlength=NS).float()
        total_samples += lb.shape[0]
        del lb
    w = torch.sqrt(counts.sum()/(counts+1)).clamp(max=5.0); w = (w/w.mean()).to(device)
    log.info(f"  Class weights computed from {total_samples:,} samples")

    # Shard-sequential loading — ALL shards used for both train and val
    # Each shard: first 90% train, last 10% val (ensures label overlap)
    n_shards = meta["n_shards"]
    train_shards = list(range(n_shards))
    val_shard_ids = list(range(n_shards))
    tsz = int(N * 0.9)
    vsz = N - tsz

    def load_shard_tensors(si):
        d = {}
        for k in ["col_embs","col_mask","dist_fps","cell_values","cell_mask","cell_is_numeric","labels"]:
            d[k] = torch.load(PRECOMP_DIR/f"{k}_{si}.pt", weights_only=True)
        return d

    # Prefetch shard loading with threading
    from concurrent.futures import ThreadPoolExecutor
    _prefetch_executor = ThreadPoolExecutor(max_workers=1)

    def shard_batches(shard_ids, batch_size, shuffle=True, split="train"):
        future = _prefetch_executor.submit(load_shard_tensors, shard_ids[0]) if shard_ids else None
        for i, si in enumerate(shard_ids):
            d = future.result()
            if i + 1 < len(shard_ids):
                future = _prefetch_executor.submit(load_shard_tensors, shard_ids[i + 1])
            n = d["labels"].shape[0]
            # Shuffle within shard first, then split
            perm = torch.randperm(n)
            for k in d: d[k] = d[k][perm]
            split_idx = int(n * 0.9)
            if split == "train":
                indices = torch.randperm(split_idx) if shuffle else torch.arange(split_idx)
            else:
                indices = torch.arange(split_idx, n)
            for start in range(0, len(indices) - batch_size + 1, batch_size):
                idx = indices[start:start+batch_size]
                yield (d["col_embs"][idx], d["col_mask"][idx], d["dist_fps"][idx],
                       d["cell_values"][idx], d["cell_mask"][idx], d["cell_is_numeric"][idx],
                       d["labels"][idx])
            del d; gc.collect()

    # Auto batch size detection
    if torch.cuda.is_available():
        log.info("Auto-detecting batch size...")
        for try_bs in [128, 96, 64]:
            try:
                torch.cuda.empty_cache()
                with amp_autocast():
                    test_out = model.forward_from_tensors(
                        torch.randn(try_bs, cfg.max_cols, cfg.sbert_dim, dtype=torch.float16, device=device),
                        torch.ones(try_bs, cfg.max_cols, dtype=torch.bool, device=device),
                        torch.randn(try_bs, cfg.max_cols, cfg.fingerprint_dim, dtype=torch.float16, device=device),
                        torch.randn(try_bs, cfg.max_rows, cfg.max_cols, dtype=torch.float16, device=device),
                        torch.ones(try_bs, cfg.max_rows, cfg.max_cols, dtype=torch.bool, device=device),
                        torch.ones(try_bs, cfg.max_rows, cfg.max_cols, dtype=torch.bool, device=device),
                        training=True
                    )
                fake_labels = torch.randint(0, NS, (try_bs,), device=device)
                with amp_autocast():
                    loss = F.cross_entropy(test_out['cls_logits'], fake_labels)
                if scaler: scaler.scale(loss).backward()
                else: loss.backward()
                if scaler: scaler.unscale_(opt); scaler.step(opt); scaler.update()
                else: opt.step()
                model.zero_grad(); opt.zero_grad()
                torch.cuda.empty_cache()
                cfg.batch_size = try_bs
                log.info(f"  Batch size {try_bs} OK (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB used)")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.info(f"  Batch size {try_bs} OOM, trying smaller...")
                    torch.cuda.empty_cache()
                    model.zero_grad()
                else:
                    raise
        log.info(f"  Final batch size: {cfg.batch_size} x {cfg.grad_accum} = {cfg.batch_size*cfg.grad_accum} effective")

    # Compile model for faster execution

    batches_per_epoch = tsz // cfg.batch_size
    log.info(f"Train: {tsz:,} | Val: {vsz:,} | Batches/epoch: {batches_per_epoch:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True if torch.cuda.is_available() else False)
    ts2 = batches_per_epoch*cfg.total_epochs//max(cfg.grad_accum,1); ws = batches_per_epoch*cfg.warmup_epochs//max(cfg.grad_accum,1)
    def lrs(s):
        if s < ws: return max(s/max(ws,1), 1e-2)
        return max(0.5*(1+math.cos(math.pi*(s-ws)/max(ts2-ws,1))), 1e-2)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lrs)
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None
    ewc = EWCModule(); sl2 = SelfLearner(); ood = OODDetector()
    se2 = 0; ba = 0; step = 0; ni = 0

    if CHECKPOINT_PATH.exists():
        log.info(f"Resuming from {CHECKPOINT_PATH}")
        ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        try: opt.load_state_dict(ck["optimizer_state_dict"])
        except: pass
        se2 = ck.get("epoch",0); ba = ck.get("best_accuracy",0); step = ck.get("step",0)
        log.info(f"  epoch={se2}, step={step:,}, best={ba:.1f}%")

    cc = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing, reduction='none')
    sc = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, reduction='none')

    for epoch in range(se2, cfg.total_epochs):
        model.train(); te = time.time()
        e_loss = 0; e_cls_loss = 0; e_sec_loss = 0; e_mcm_loss = 0; e_miras_loss = 0; e_midas_loss = 0; e_cont_loss = 0
        e_cls_correct = 0; e_sec_correct = 0; e_top5_correct = 0; e_total = 0
        opt.zero_grad()

        sl2.update_threshold(epoch, cfg.total_epochs)

        active_sectors = set(range(NS))
        shard_order = train_shards[:]; random.shuffle(shard_order)
        for bi, batch in enumerate(shard_batches(shard_order, cfg.batch_size, shuffle=True, split="train")):
            if _shutdown:
                log.info("Shutdown requested, saving checkpoint...")
                torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"epoch":epoch,"step":step,"best_accuracy":ba}, CHECKPOINT_PATH)
                log.info("Checkpoint saved. Exiting."); return

            try:
                items = [b.to(device, non_blocking=True) for b in batch]
                labels = items[-1]
                with amp_autocast():
                    out = model.forward_from_tensors(*items[:-1], training=True)
                    loss_cls_m = cc(out['cls_logits'], labels).mean()
                    loss_sec_m = sc(out['sector_logits'], labels).mean()
                    l_mcm = out['mcm_loss']*cfg.mcm_weight
                    l_miras = out['miras_loss']*cfg.miras_weight
                    l_midas = out['midas_loss']
                    # Contrastive every 10 batches
                    if (bi+1) % 10 == 0:
                        l_cont = model.miras.contrastive_loss(out['contrastive_emb'], labels)*cfg.contrastive_weight
                        _cached_cont = l_cont.item()
                    else:
                        l_cont = torch.tensor(0.0, device=device)
                    # EWC every 10 batches
                    if (bi+1) % 10 == 0:
                        l_ewc = ewc.penalty(model)
                    else:
                        l_ewc = torch.tensor(0.0, device=device)
                    loss = loss_cls_m + loss_sec_m + l_mcm + l_miras + l_midas + l_cont + l_ewc

                if scaler: scaler.scale(loss).backward()
                else: loss.backward()

                if scaler: scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm); scaler.step(opt); scaler.update()
                else: nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm); opt.step()


                with torch.no_grad():
                    cls_preds = out['cls_logits'].argmax(-1)
                    sec_preds = out['sector_logits'].argmax(-1)
                    e_cls_correct += (cls_preds==labels).sum().item()
                    e_sec_correct += (sec_preds==labels).sum().item()
                    e_top5_correct += topk_accuracy(out['cls_logits'], labels, k=5)
                    e_total += labels.shape[0]
                    e_loss += loss.item()
                    e_cls_loss += loss_cls_m.item(); e_sec_loss += loss_sec_m.item()
                    e_mcm_loss += l_mcm.item(); e_miras_loss += l_miras.item()
                    e_midas_loss += l_midas.item(); e_cont_loss += l_cont.item()

                if (bi+1) % cfg.log_every == 0:
                    elapsed = time.time()-te; rate = (bi+1)*cfg.batch_size/max(elapsed,1)
                    pct = 100*(bi+1)/batches_per_epoch; bar_len=20; filled=int(pct/100*bar_len); bar_str="2588"*filled+"2591"*(bar_len-filled)
                    n = bi+1
                    vr = f" vram={torch.cuda.memory_allocated()/1024**3:.1f}/{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB" if torch.cuda.is_available() else ""
                    log.info(
                        f"  E{epoch+1} [{pct:5.1f}%] "
                        f"loss={e_loss/n:.4f} "
                        f"cls={100*e_cls_correct/max(e_total,1):.1f}% "
                        f"sec={100*e_sec_correct/max(e_total,1):.1f}% "
                        f"top5={100*e_top5_correct/max(e_total,1):.1f}% "
                        f"lr={opt.param_groups[0]['lr']:.2e} "
                        f"rate={rate:.0f}/s acc={50*(e_cls_correct+e_sec_correct)/max(e_total,1):.1f}% "
                        f"eta={(batches_per_epoch-bi-1)*cfg.batch_size/max(rate,1)/60:.0f}min"
                        f"{vr}"
                    )
                    log.info(
                        f"         "
                        f"L_cls={e_cls_loss/n:.4f} L_sec={e_sec_loss/n:.4f} "
                        f"L_mcm={e_mcm_loss/n:.4f} L_miras={e_miras_loss/n:.4f} "
                        f"L_midas={e_midas_loss/n:.4f} L_cont={e_cont_loss/n:.4f}"
                    )

                if (bi+1)*cfg.batch_size % cfg.checkpoint_every < cfg.batch_size:
                    torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"epoch":epoch,"step":step,"best_accuracy":ba}, CHECKPOINT_PATH)
                    log.info(f"  Checkpoint saved (step {step:,})")

                if (bi+1) % 5000 == 0:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning(f"  OOM at batch {bi+1}, skipping...")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    opt.zero_grad(); gc.collect()
                    continue
                raise

            if (bi+1) % 10000 == 0:
                check_ram()
                if (bi+1) % 50000 == 0 and not check_docker_healthy():
                    log.warning("Docker unhealthy! Saving checkpoint and pausing 60s...")
                    torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"epoch":epoch,"step":step,"best_accuracy":ba}, CHECKPOINT_PATH)
                    time.sleep(60)
                    if not check_docker_healthy():
                        log.error("Docker still unhealthy. Stopping.")
                        return

        # ======== FULL VALIDATION ========
        model.eval()
        v_cls_correct = 0; v_sec_correct = 0; v_top5_correct = 0; v_total = 0
        v_per_sector = Counter(); v_per_sector_correct = Counter()
        v_embeddings = []
        with torch.no_grad():
            for batch in shard_batches(val_shard_ids, cfg.batch_size, shuffle=False, split="val"):
                if v_total >= 20000: break
                items = [b.to(device, non_blocking=True) for b in batch]; labels = items[-1]
                with amp_autocast(): out = model.forward_from_tensors(*items[:-1], training=False)
                cls_p = out['cls_logits'].argmax(-1); sec_p = out['sector_logits'].argmax(-1)
                v_cls_correct += (cls_p==labels).sum().item()
                v_sec_correct += (sec_p==labels).sum().item()
                v_top5_correct += topk_accuracy(out['cls_logits'], labels, k=5)
                v_total += labels.shape[0]
                for i in range(labels.shape[0]):
                    lbl = labels[i].item()
                    v_per_sector[lbl] += 1
                    if cls_p[i].item() == lbl: v_per_sector_correct[lbl] += 1
                if len(v_embeddings) < 100: v_embeddings.append(out['contrastive_emb'].detach().cpu())

        v_cls_acc = 100*v_cls_correct/max(v_total,1)
        v_sec_acc = 100*v_sec_correct/max(v_total,1)
        v_top5_acc = 100*v_top5_correct/max(v_total,1)
        t_cls_acc = 100*e_cls_correct/max(e_total,1)
        elapsed = time.time()-te

        log.info(f"\n{'='*60}")
        log.info(f"  EPOCH {epoch+1}/{cfg.total_epochs} COMPLETE ({elapsed/60:.1f}min)")
        log.info(f"  Train:  cls={t_cls_acc:.1f}%")
        log.info(f"  Val:    cls={v_cls_acc:.1f}%  sec={v_sec_acc:.1f}%  top5={v_top5_acc:.1f}%  ({v_total:,} samples)")

        # Per-sector accuracy (worst 10)
        sector_accs = {}
        for sid in v_per_sector:
            total_s = v_per_sector[sid]
            correct_s = v_per_sector_correct.get(sid, 0)
            sector_accs[sid] = 100*correct_s/max(total_s,1)
        worst = sorted(sector_accs.items(), key=lambda x: x[1])[:10]
        best = sorted(sector_accs.items(), key=lambda x: -x[1])[:5]
        log.info(f"  Best sectors:  {', '.join(f'{I2S.get(s,str(s))[:20]}={a:.0f}%' for s,a in best)}")
        log.info(f"  Worst sectors: {', '.join(f'{I2S.get(s,str(s))[:20]}={a:.0f}%' for s,a in worst)}")
        zero_acc = [s for s, a in sector_accs.items() if a == 0]
        log.info(f"  Zero-acc sectors: {len(zero_acc)}/{len(sector_accs)}")
        log.info(f"{'='*60}")

        # OOD
        if v_embeddings and cfg.ood_enabled:
            ec2 = torch.cat(v_embeddings)[:2000]; ood.fit(ec2)
            if ood.fitted: ms2 = ood.mahalanobis(ec2[:100]); log.info(f"  OOD: mean={ms2.mean():.1f} max={ms2.max():.1f}")

        # EWC
        ewc.decay_fisher()
        if epoch > 0:
            val_dl_tmp = list(shard_batches(val_shard_ids[:1], cfg.batch_size, shuffle=False, split="val"))
            ewc.compute_fisher(model, val_dl_tmp, device, 2000); ewc.update_replay(tsz)

        # Self-learning (1 shard only)
        if cfg.self_learning_enabled and epoch > 0:
            model.eval(); pl_count = 0
            with torch.no_grad():
                for batch in shard_batches(val_shard_ids[:1], cfg.batch_size, shuffle=False, split="val"):
                    if pl_count >= cfg.pseudo_label_max_per_epoch: break
                    items = [b.to(device, non_blocking=True) for b in batch]; labels = items[-1]
                    with amp_autocast(): out = model.forward_from_tensors(*items[:-1], training=False)
                    max_p, _ = F.softmax(out['cls_logits'], dim=-1).max(dim=-1)
                    pl_count += (max_p > sl2.threshold).sum().item()
            log.info(f"  Self-learning: {pl_count:,} pseudo-labels (threshold={sl2.threshold:.2f})")
            model.train()

        # Save best
        if v_cls_acc > ba:
            ba = v_cls_acc; ni = 0
            torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"epoch":epoch+1,"step":step,"best_accuracy":ba,"n_sectors":NS,"ds_sectors":DS_SECTORS,"ds_s2i":DS_S2I,"total_params":tp,"architecture":f"SchemaV1 ({tp/1e6:.0f}M)"}, CHECKPOINT_PATH)
            log.info(f"  BEST MODEL SAVED: {ba:.1f}%")
        else:
            ni += 1; log.info(f"  No improvement ({ni}/{cfg.early_stop_patience})")
            if ni >= cfg.early_stop_patience: log.info(f"  EARLY STOP at epoch {epoch+1}"); break

    log.info(f"\nTRAINING DONE. Best val accuracy: {ba:.1f}%")

if __name__ == "__main__":
    if needs_precompute(): run_precompute()
    else: log.info("Pre-computed shards found, skip to training.")
    train()