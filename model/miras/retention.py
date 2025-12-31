import torch
import torch.nn as nn

class AdaptiveRetention(nn.Module):
    def __init__(self, d_model, sector='default'):
        super().__init__()
        self.sector = sector
        self.alpha = nn.Parameter(torch.ones(d_model) * self.get_alpha(sector))
        
    def get_alpha(self, sector):
        alphas = {
            'stock': 0.5,
            'crypto': 0.4,
            'weather': 0.6,
            'zoning': 0.9,
            'healthcare': 0.85,
            'default': 0.7
        }
        return alphas.get(sector, 0.7)
    
    def forward(self, memory_current, memory_prev):
        return self.alpha * memory_prev + (1 - self.alpha) * memory_current
