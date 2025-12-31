import torch
import torch.nn as nn

class SchemaProcessing(nn.Module):
    def __init__(self, d_model, n_heads=8, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads,
                dim_feedforward=d_model*2,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, schema_info):
        x = self.proj(schema_info)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
