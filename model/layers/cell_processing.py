import torch
import torch.nn as nn

class CellProcessing(nn.Module):
    def __init__(self, d_model, vocab_size=50000, n_types=10, max_cols=64):
        super().__init__()
        self.d_model = d_model
        self.continuous_proj = nn.Linear(1, d_model)
        self.col_embed = nn.Embedding(max_cols, d_model)
        self.pos_embed = nn.Embedding(max_cols, d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        # RMSNorm yerine basit scale
        self.scale = nn.Parameter(torch.ones(d_model) * 0.1)
        
    def forward(self, values, types=None, continuous=False):
        if continuous or values.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            return self.forward_continuous(values)
        return self.forward_continuous(values)
    
    def forward_continuous(self, values):
        batch, n_cols = values.shape
        
        # Value embedding - SCALE ETME, orijinal değeri koru
        values_expanded = values.unsqueeze(-1)
        value_emb = self.continuous_proj(values_expanded)
        
        col_ids = torch.arange(n_cols, device=values.device)
        col_emb = self.col_embed(col_ids).unsqueeze(0).expand(batch, -1, -1)
        pos_emb = self.pos_embed(col_ids).unsqueeze(0).expand(batch, -1, -1)
        
        fused = self.fusion(torch.cat([value_emb, col_emb, pos_emb], dim=-1))
        
        # Normalize etme, sadece scale
        return fused * self.scale
