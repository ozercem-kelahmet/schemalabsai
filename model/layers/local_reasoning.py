import torch
import torch.nn as nn

class LocalReasoningLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + 0.1 * attn_out  # Sabit 0.1
        x = x + 0.1 * self.mlp(x)
        return x
