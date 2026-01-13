import torch
import torch.nn as nn

class RowWiseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class ColumnWiseAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_cols=64):
        super().__init__()
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class AxialEncoder(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.row_attn = RowWiseAttention(d_model, n_heads)
        self.col_attn = ColumnWiseAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.row_attn(x)
        x = self.col_attn(x)
        x = self.norm(x + self.mlp(x))
        return x


class LocalReasoningLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            AxialEncoder(d_model, n_heads) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
