import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint as grad_checkpoint

class SchemaV1Config:
    d_model = 512
    n_heads = 8
    head_dim = 64
    n_latent = 128
    ffn_hidden = 2048
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
    n_sectors = 10000

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

cfg = SchemaV1Config()
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
            noisy = (current + torch.randn_like(current) * cfg.midas_noise_std * work_mask.unsqueeze(-1).float()) if (training and it == 0) else current
            val_emb = self.value_proj(noisy * work_mask.unsqueeze(-1).float())
            mask_emb = self.mask_proj(work_mask.unsqueeze(-1).float().expand_as(noisy))
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
class SinusoidalPE(nn.Module):
    def __init__(self, max_len, d):
        super().__init__()
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe)
    def forward(self, n):
        return self.pe[:n]

class CellProcessing(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.sbert_proj = nn.Sequential(nn.Linear(cfg.sbert_dim, d), RMSNorm(d))
        self.fp_proj = nn.Sequential(nn.Linear(cfg.fingerprint_dim, d), RMSNorm(d))
        self.value_proj = nn.Linear(1, d)
        self.type_emb = nn.Embedding(3, d)
        self.pos_emb = SinusoidalPE(10000, d)
        self.row_pos_emb = SinusoidalPE(10000, d)
        self.fusion_col = nn.Sequential(nn.Linear(d*3, d), RMSNorm(d), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), RMSNorm(d))
        self.fusion_cell = nn.Sequential(nn.Linear(d*4, d), RMSNorm(d), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), RMSNorm(d))

    def forward(self, col_embs, dist_fps, cell_values, cell_mask, cell_is_numeric):
        B, C, _ = col_embs.shape; R = cell_values.shape[1]
        sbert_emb = self.sbert_proj(col_embs.float())
        fp_emb = self.fp_proj(dist_fps.float())
        pos_emb = self.pos_emb(C).to(col_embs.device).unsqueeze(0).expand(B, -1, -1)
        col_features = self.fusion_col(torch.cat([sbert_emb, fp_emb, pos_emb], dim=-1))
        row_emb = self.row_pos_emb(R).to(col_embs.device).unsqueeze(0).unsqueeze(2).expand(B, -1, C, -1)
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
        self.norm1 = RMSNorm(d); self.attn = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.norm2 = RMSNorm(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x, mask=None):
        h = self.norm1(x); h, _ = self.attn(h, h, h, key_padding_mask=mask); x = x + self.drop(h)
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
        self.rn = RMSNorm(d); self.ra = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.cn = RMSNorm(d); self.ca = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.fn = RMSNorm(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        B, R, C, D = x.shape
        xr = x.reshape(B*R, C, D); h = self.rn(xr); h, _ = self.ra(h, h, h); xr = xr + self.drop(h); x = xr.reshape(B, R, C, D)
        xc = x.permute(0,2,1,3).reshape(B*C, R, D); h = self.cn(xc); h, _ = self.ca(h, h, h); xc = xc + self.drop(h); x = xc.reshape(B, C, R, D).permute(0,2,1,3)
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
        self.cnq = RMSNorm(d); self.cnkv = RMSNorm(d); self.ca = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.sn = RMSNorm(d); self.sa = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True)
        self.fn = RMSNorm(d); self.ffn = SwiGLU(d, cfg.ffn_hidden); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, lat, ctx):
        h = self.cnq(lat); c = self.cnkv(ctx); h, _ = self.ca(h, c, c); lat = lat + self.drop(h)
        h = self.sn(lat); h, _ = self.sa(h, h, h); lat = lat + self.drop(h)
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
        self.ca = nn.MultiheadAttention(d, nh, dropout=cfg.dropout, batch_first=True); self.drop = nn.Dropout(cfg.dropout)
    def forward(self, cr, se):
        q = self.nq(cr); kv = self.nkv(se); h, _ = self.ca(q, kv, kv); return cr + self.drop(h)

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
        pos = (es*mp.float()).sum(1); neg = (es*mn.float()).sum(1)
        return (-torch.log(pos/(pos+neg+1e-8)+1e-8)).mean()

    def calibrate(self, lo): return self.layers[0].calibrated_logits(lo)

# ============================================================
# EWC + OOD + CURRICULUM + SELF-LEARNING
# ============================================================


class SchemaV1(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = SchemaV1Config()
        self.cfg = cfg
        d = cfg.d_model
        self.d = d
        self.midas = MIDAS(d)
        self.cell_proc = CellProcessing(d)
        self.schema_proc = SchemaProcessing(d, cfg.n_heads, cfg.n_schema_layers)
        self.local_reason = LocalReasoning(d, cfg.n_heads, cfg.n_local_layers)
        self.global_reason = GlobalReasoning(d, cfg.n_heads, cfg.n_global_layers, cfg.n_latent)
        self.sector_head = SectorHead(d)
        self.cls_head = ClassificationHead(d, cfg.n_sectors)
        self.reg_head = RegressionHead(d)
        self.mcm_head = MCMHead(d)
        self.domain_adapter = DomainSchemaAdapter(d)
        self.domain_inject = DomainKnowledgeInjection(d, cfg.n_heads)
        self.schema_cell_fusion = SchemaCellFusion(d)
        self.domain_heads = DomainSpecificHeads(d)
        self.miras = MIRAS(d, cfg.n_miras_layers)
        self.combine_proj = nn.Sequential(nn.Linear(d*3, d), RMSNorm(d))
        self.register_buffer('_sem', torch.zeros(cfg.n_sectors, cfg.sbert_dim))

    def set_sector_emb(self, m):
        self._sem = m

    def forward_from_tensors(self, ce, cm, df, cv, cmask, cin, training=True):
        B = ce.shape[0]
        col_feat, cell_grid = self.cell_proc(ce, df, cv, cmask, cin)
        midas_out, rl, il, confs = self.midas(col_feat, cm, training)
        schema_out = self.schema_proc(midas_out, cm)
        gc = schema_out.mean(dim=1)
        sa = self.domain_adapter(schema_out, gc)
        cf = cell_grid.mean(dim=1)
        ci = self.domain_inject(cf, sa)
        sr = sa.mean(dim=1)
        cr = ci.mean(dim=1)
        fused = self.schema_cell_fusion(sr, cr)
        lo = self.local_reason(cell_grid)
        go = self.global_reason(sa)
        combined = self.combine_proj(torch.cat([lo, go, fused], dim=-1))
        combined, ms, ml = self.miras(combined, None)
        combined = self.domain_heads(combined)
        sl = self.sector_head(combined, self._sem)
        cl = self.miras.calibrate(self.cls_head(combined))
        rv = self.reg_head(combined)
        mcml = torch.tensor(0.0, device=ce.device)
        if training:
            mm = (torch.rand(B, self.cfg.max_cols, device=ce.device) < self.cfg.mcm_mask_ratio) & cm
            if mm.any():
                mp = self.mcm_head(midas_out)
                tv = df[:,:,0].float()
                mcml = F.mse_loss(mp[mm], tv[mm])
        return {
            'sector_logits': sl,
            'cls_logits': cl,
            'reg_output': rv,
            'embeddings': combined,
            'midas_loss': self.cfg.midas_recon_weight*rl + self.cfg.midas_imputation_weight*il,
            'mcm_loss': mcml,
            'miras_loss': ml,
            'contrastive_emb': combined
        }

    def count_params(self):
        t = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return t, tr


class SchemaV1Adapter:
    def __init__(self, cfg=None, sbert_model=None, device='cpu'):
        self.cfg = cfg or SchemaV1Config()
        self.device = device
        self._sbert = sbert_model
    
    def _get_sbert(self):
        if self._sbert is None:
            import logging
            logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
            logging.getLogger('transformers').setLevel(logging.ERROR)
            import warnings
            warnings.filterwarnings('ignore')
            import os as _os
            _os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            _os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        return self._sbert
    
    def _signed_log_np(self, x):
        import numpy as np
        s = np.sign(x); ax = np.abs(x)
        return np.where(ax > 1, s * np.log1p(ax) / 20.0, x).astype(np.float32)
    
    def _compute_fingerprint(self, col_values_np):
        import numpy as np
        col = col_values_np.astype(np.float64)
        n = len(col)
        fp = np.array([
            float(col.mean()) if n else 0.0,
            float(col.std()) if n else 0.0,
            float(col.min()) if n else 0.0,
            float(col.max()) if n else 0.0,
            1.0,
            float(len(np.unique(col))) / max(n, 1),
            float(n),
        ], dtype=np.float32)
        return self._signed_log_np(fp)
    
    def df_to_tensors(self, df, batch_size=1):
        import numpy as np
        import pandas as pd
        import torch
        
        max_cols = self.cfg.max_cols
        max_rows = self.cfg.max_rows
        sbert_dim = self.cfg.sbert_dim
        fp_dim = self.cfg.fingerprint_dim
        
        cols = list(df.columns)[:max_cols]
        n_cols = len(cols)
        
        sbert = self._get_sbert()
        col_names_norm = [c.lower().replace("_", " ") for c in cols]
        col_embs_np = sbert.encode(col_names_norm, convert_to_numpy=True, show_progress_bar=False)
        
        ce = torch.zeros(1, max_cols, sbert_dim, dtype=torch.float32)
        ce[0, :n_cols] = torch.from_numpy(col_embs_np).float()
        
        cm = torch.zeros(1, max_cols, dtype=torch.bool)
        cm[0, :n_cols] = True
        
        df_tensor = torch.zeros(1, max_cols, fp_dim, dtype=torch.float32)
        
        sample_rows = min(len(df), max_rows)
        cv = torch.zeros(1, max_rows, max_cols, dtype=torch.float32)
        cmask = torch.zeros(1, max_rows, max_cols, dtype=torch.bool)
        cin = torch.zeros(1, max_rows, max_cols, dtype=torch.bool)
        
        numeric_cols = []
        for i, c in enumerate(cols):
            series = df[c]
            numeric = pd.to_numeric(series, errors='coerce')
            valid_ratio = numeric.notna().sum() / max(len(series), 1)
            if valid_ratio >= 0.5:
                col_values = numeric.fillna(0).values.astype(np.float32)
            else:
                codes = pd.Categorical(series.astype(str).fillna("")).codes
                col_values = codes.astype(np.float32)
            numeric_cols.append(col_values)
        
        for i in range(len(cols)):
            full_col = numeric_cols[i]
            full_col_log = self._signed_log_np(full_col)
            fp_vec = self._compute_fingerprint(full_col_log)
            df_tensor[0, i] = torch.from_numpy(fp_vec).float()
            col_vals_log = full_col_log[:sample_rows]
            for r in range(sample_rows):
                cv[0, r, i] = float(col_vals_log[r])
                cin[0, r, i] = True
        
        return ce.to(self.device), cm.to(self.device), df_tensor.to(self.device), cv.to(self.device), cmask.to(self.device), cin.to(self.device)


class SchemaV1FineTuneHead(nn.Module):
    def __init__(self, d_model, n_classes, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        if task_type == 'classification':
            self.head = nn.Linear(d_model, n_classes)
        else:
            self.head = nn.Linear(d_model, 1)
    
    def forward(self, embeddings):
        h = self.dropout(self.norm(embeddings))
        return self.head(h)


class SchemaV1FineTuneWrapper(nn.Module):
    def __init__(self, base_schema_v1, n_classes, task_type='classification', freeze_backbone=True):
        super().__init__()
        self.backbone = base_schema_v1
        self.task_type = task_type
        self.n_classes = n_classes
        self.ft_head = SchemaV1FineTuneHead(base_schema_v1.d, n_classes, task_type)
        self._backbone_frozen = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def unfreeze_backbone(self):
        self._backbone_frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True
    
    def compute_embeddings(self, ce, cm, df, cv, cmask, cin):
        with torch.no_grad():
            out = self.backbone.forward_from_tensors(ce, cm, df, cv, cmask, cin, training=False)
        emb = out['embeddings'].mean(dim=1) if out['embeddings'].dim() == 3 else out['embeddings']
        return emb
    
    def forward(self, ce, cm=None, df=None, cv=None, cmask=None, cin=None):
        if cm is None:
            return self.ft_head(ce)
        with torch.set_grad_enabled(not self._backbone_frozen):
            out = self.backbone.forward_from_tensors(ce, cm, df, cv, cmask, cin, training=False)
        emb = out['embeddings'].mean(dim=1) if out['embeddings'].dim() == 3 else out['embeddings']
        logits = self.ft_head(emb)
        return logits


def detect_task_type(df, target_col):
    import pandas as pd
    import numpy as np
    series = df[target_col].dropna()
    if len(series) == 0:
        return 'classification', 2
    numeric = pd.to_numeric(series, errors='coerce').dropna()
    numeric_ratio = len(numeric) / len(series)
    if numeric_ratio < 0.95:
        unique_vals = series.unique()
        return 'classification', len(unique_vals)
    unique_numeric = numeric.unique()
    if len(unique_numeric) <= 20 and all(float(v).is_integer() for v in unique_numeric if not np.isnan(v)):
        return 'classification', len(unique_numeric)
    return 'regression', 1


class SchemaV1EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def precompute_embeddings(base_model, df, target_col, adapter, label_encoder=None, device='cpu', batch_size=64):
    import hashlib, os, pickle, pandas as pd
    cache_dir = '/tmp/schemalabs_emb_cache'
    os.makedirs(cache_dir, exist_ok=True)
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"emb_{df_hash}_{len(df)}_{len(df.columns)}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as cf:
                cached = pickle.load(cf)
            print(f"[SchemaV1] Embedding cache HIT: {cache_path}")
            return cached
        except:
            pass
    result = _precompute_embeddings_inner(base_model, df, target_col, adapter, label_encoder=label_encoder, device=device, batch_size=batch_size)
    try:
        with open(cache_path, 'wb') as cf:
            pickle.dump(result, cf)
        print(f"[SchemaV1] Embedding cache SAVED: {cache_path}")
    except Exception as e:
        print(f"[SchemaV1] Cache save failed: {e}")
    return result


def _precompute_embeddings_inner(base_model, df, target_col, adapter, label_encoder=None, device='cpu', batch_size=64):
    import pandas as pd
    import numpy as np
    import time
    
    t0 = time.time()
    ds = SchemaV1PrecomputedDataset(df, target_col, base_model.cfg, adapter, label_encoder=label_encoder)
    print(f"[SchemaV1] Tensor precompute: {time.time()-t0:.1f}s")
    
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_embs = []
    all_labels = []
    base_model.eval()
    use_half = next(base_model.parameters()).device.type == "cuda"
    if use_half:
        base_model.half()
    n_batches = (len(ds) + batch_size - 1) // batch_size
    t1 = time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            ce, cm, dfp, cv, cmask, cin, y = batch
            ce = ce.to(device); cm = cm.to(device); dfp = dfp.to(device)
            cv = cv.to(device); cmask = cmask.to(device); cin = cin.to(device)
            if use_half:
                ce = ce.half(); cm = cm.half(); dfp = dfp.half(); cv = cv.half()
            out = base_model.forward_from_tensors(ce, cm, dfp, cv, cmask, cin, training=False)
            emb = out['embeddings'].mean(dim=1) if out['embeddings'].dim() == 3 else out['embeddings']
            all_embs.append(emb.cpu())
            all_labels.append(y)
            if bi % 10 == 0:
                elapsed = time.time() - t1
                eta = elapsed / (bi+1) * (n_batches - bi - 1)
                print(f"[SchemaV1] Backbone forward {bi+1}/{n_batches} ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")
    
    if use_half:
        base_model.float()
    embeddings = torch.cat(all_embs, dim=0).float()
    labels = torch.cat(all_labels, dim=0)
    print(f"[SchemaV1] Precomputed embeddings: {tuple(embeddings.shape)} in {time.time()-t0:.0f}s total")
    return SchemaV1EmbeddingDataset(embeddings, labels)


class SchemaV1PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, df, target_col, cfg, adapter, label_encoder=None):
        import pandas as pd
        import numpy as np
        self.cfg = cfg
        self.adapter = adapter
        self.label_encoder = label_encoder
        
        feature_df = df.drop(columns=[target_col])
        y_series = df[target_col]
        
        if label_encoder is not None:
            self.y = label_encoder.transform(y_series.astype(str))
            self.y = torch.tensor(self.y, dtype=torch.long)
        else:
            self.y = torch.tensor(pd.to_numeric(y_series, errors='coerce').fillna(0).values, dtype=torch.float32)
        
        print(f"[SchemaV1Dataset] Precomputing tensors for {len(feature_df)} rows...")
        chunk_df = feature_df.copy()
        ce, cm, dfp, cv, cmask, cin = adapter.df_to_tensors(chunk_df)
        self.ce_shared = ce[0]
        self.cm_shared = cm[0]
        self.dfp_shared = dfp[0]
        
        max_rows = cfg.max_rows
        max_cols = cfg.max_cols
        n = len(feature_df)
        
        self.cv_all = torch.zeros(n, max_rows, max_cols, dtype=torch.float32)
        self.cmask_all = torch.zeros(n, max_rows, max_cols, dtype=torch.bool)
        self.cin_all = torch.zeros(n, max_rows, max_cols, dtype=torch.bool)
        
        cols = list(feature_df.columns)[:max_cols]
        n_cols = len(cols)
        numeric_np = feature_df[cols].apply(lambda c: pd.to_numeric(c, errors='coerce')).values.astype(np.float32)
        
        missing_mask = np.isnan(numeric_np) | np.isinf(numeric_np)
        numeric_np_clean = np.nan_to_num(numeric_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.cv_all[:, 0, :n_cols] = torch.from_numpy(numeric_np_clean)
        self.cmask_all[:, 0, :n_cols] = torch.from_numpy(missing_mask)
        self.cin_all[:, 0, :n_cols] = torch.from_numpy(~missing_mask)
        
        print(f"[SchemaV1Dataset] Done. cv shape: {tuple(self.cv_all.shape)}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.ce_shared,
            self.cm_shared,
            self.dfp_shared,
            self.cv_all[idx],
            self.cmask_all[idx],
            self.cin_all[idx],
            self.y[idx]
        )
