import torch
import torch.nn as nn
import math

class CellProcessing(nn.Module):
    def __init__(self, d_model, vocab_size=50000, n_types=10, max_cols=64):
        super().__init__()
        self.d_model = d_model
        self.max_cols = max_cols
        self.continuous_proj = nn.Linear(1, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.col_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def get_sinusoidal_encoding(self, n_cols, device):
        position = torch.arange(n_cols, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * 
                           (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(n_cols, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        return pe
        
    def forward(self, values, cat_indices=None, types=None, continuous=False):
        batch, n_cols = values.shape
        device = values.device
        
        values_expanded = values.unsqueeze(-1)
        value_emb = self.continuous_proj(values_expanded)
        
        if cat_indices is not None and len(cat_indices) > 0:
            for idx in cat_indices:
                if idx < n_cols:
                    cat_emb = self.token_embed(values[:, idx].long())
                    value_emb[:, idx, :] = cat_emb
        
        pos_emb = self.get_sinusoidal_encoding(n_cols, device).unsqueeze(0).expand(batch, -1, -1)
        col_pos = (torch.arange(n_cols, dtype=torch.float, device=device) / max(n_cols - 1, 1)).unsqueeze(-1)
        col_emb = self.col_proj(col_pos).unsqueeze(0).expand(batch, -1, -1)
        
        fused = self.fusion(torch.cat([value_emb, col_emb, pos_emb], dim=-1))
        return self.norm(fused)
