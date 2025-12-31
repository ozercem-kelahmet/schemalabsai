import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * 4)
        self.w2 = nn.Linear(d_model, d_model * 4)
        
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

class MLPMemory(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * expansion)
        self.activation = SwiGLU(d_model)
        self.w2 = nn.Linear(d_model * expansion, d_model)
        self.norm = RMSNorm(d_model)
        
    def forward(self, x):
        residual = x
        h = self.w1(x)
        h = self.activation(x)
        h = self.w2(h)
        return self.norm(residual + h)
