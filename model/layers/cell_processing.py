import torch
import torch.nn as nn

class CellProcessing(nn.Module):
    def __init__(self, d_model, vocab_size=50000, n_types=10, max_cols=64):
        super().__init__()
        self.d_model = d_model
        # Sayısal değerler için
        self.continuous_proj = nn.Linear(1, d_model)
        # Kategorik değerler için
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # Kolon ve pozisyon embedding
        self.col_embed = nn.Embedding(max_cols, d_model)
        self.pos_embed = nn.Embedding(max_cols, d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, values, cat_indices=None, types=None, continuous=False):
        """
        Args:
            values: (batch, n_cols) - tüm değerler (sayısal + kategorik encoded)
            cat_indices: list - kategorik kolonların indeksleri [2, 3] gibi
            types: unused (backward compatibility)
            continuous: unused (backward compatibility)
        """
        batch, n_cols = values.shape
        
        # Sayısal embedding - tüm kolonlar için başlangıç
        values_expanded = values.unsqueeze(-1)
        value_emb = self.continuous_proj(values_expanded)
        
        # Kategorik kolonları token embedding ile değiştir
        if cat_indices is not None and len(cat_indices) > 0:
            for idx in cat_indices:
                cat_emb = self.token_embed(values[:, idx].long())
                value_emb[:, idx, :] = cat_emb
        
        # Kolon ve pozisyon embedding
        col_ids = torch.arange(n_cols, device=values.device)
        col_emb = self.col_embed(col_ids).unsqueeze(0).expand(batch, -1, -1)
        pos_emb = self.pos_embed(col_ids).unsqueeze(0).expand(batch, -1, -1)
        
        # Fusion
        fused = self.fusion(torch.cat([value_emb, col_emb, pos_emb], dim=-1))
        
        return self.norm(fused)
