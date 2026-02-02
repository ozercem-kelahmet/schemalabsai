import torch
import torch.nn as nn
import math

class CellProcessingAgnostic(nn.Module):
    """Feature-agnostic CellProcessing - herhangi feature sayısı ile çalışır"""
    def __init__(self, d_model, vocab_size=50000, n_types=10, max_cols=None):
        super().__init__()
        self.d_model = d_model
        
        # Sayısal değerler için
        self.continuous_proj = nn.Linear(1, d_model)
        
        # Kategorik değerler için
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding - sinusoidal (herhangi uzunluk için çalışır)
        # Embedding yerine sinusoidal kullanıyoruz
        
        # Column type encoding - learnable ama sabit boyutlu değil
        self.col_proj = nn.Linear(1, d_model)  # Column index -> embedding
        
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def get_positional_encoding(self, n_cols, device):
        """Sinusoidal positional encoding - herhangi uzunluk için"""
        position = torch.arange(n_cols, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * 
                           (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(n_cols, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.d_model//2] if self.d_model % 2 == 1 else div_term)
        return pe
    
    def forward(self, values, cat_indices=None, types=None, continuous=False):
        """
        Args:
            values: (batch, n_cols) - herhangi feature sayısı
        """
        batch, n_cols = values.shape
        device = values.device
        
        # Sayısal embedding
        values_expanded = values.unsqueeze(-1)
        value_emb = self.continuous_proj(values_expanded)
        
        # Kategorik kolonları token embedding ile değiştir
        if cat_indices is not None and len(cat_indices) > 0:
            for idx in cat_indices:
                if idx < n_cols:
                    cat_emb = self.token_embed(values[:, idx].long())
                    value_emb[:, idx, :] = cat_emb
        
        # Positional encoding - sinusoidal
        pos_emb = self.get_positional_encoding(n_cols, device).unsqueeze(0).expand(batch, -1, -1)
        
        # Column encoding - normalized index
        col_ids = torch.arange(n_cols, dtype=torch.float, device=device) / max(n_cols, 1)
        col_emb = self.col_proj(col_ids.unsqueeze(-1)).unsqueeze(0).expand(batch, -1, -1)
        
        # Fusion
        fused = self.fusion(torch.cat([value_emb, col_emb, pos_emb], dim=-1))
        
        return self.norm(fused)
