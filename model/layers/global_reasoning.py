import torch
import torch.nn as nn

class LatentTokens(nn.Module):
    """Öğrenilebilir latent vektörler"""
    def __init__(self, d_model, n_latents):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        
    def forward(self, batch_size):
        return self.latents.expand(batch_size, -1, -1)


class CrossAttention(nn.Module):
    """Cell grid ile latent tokens arası attention"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, latents, cell_grid):
        # latents query, cell_grid key/value
        attn_out, _ = self.attn(latents, cell_grid, cell_grid)
        return self.norm(latents + attn_out)


class LatentSelfAttention(nn.Module):
    """Latent tokens kendi arası attention"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class GlobalReasoningLayer(nn.Module):
    """Diagram: Latent Tokens → Cross-Attention → Latent Self-Attention → Global Latents"""
    def __init__(self, d_model, n_heads, n_latents, n_layers=2, sector='default'):
        super().__init__()
        self.latent_tokens = LatentTokens(d_model, n_latents)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            LatentSelfAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        
    def forward(self, cell_grid):
        batch_size = cell_grid.size(0)
        
        # 1. Latent tokens oluştur
        latents = self.latent_tokens(batch_size)
        
        # 2. Cross-attention + Self-attention layers
        for cross_attn, self_attn in zip(self.cross_attn_layers, self.self_attn_layers):
            latents = cross_attn(latents, cell_grid)
            latents = self_attn(latents)
        
        # 3. Global latents - pool
        global_latents = latents.mean(dim=1)
        
        return latents, global_latents
