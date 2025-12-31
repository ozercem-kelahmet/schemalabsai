import torch
import torch.nn as nn

class MomentumGD(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        self.beta = nn.Parameter(torch.ones(d_model) * 0.9)
        self.momentum = None
        
    def step(self, grad):
        if self.momentum is None:
            self.momentum = torch.zeros_like(grad)
        
        self.momentum = self.beta * self.momentum + (1 - self.beta) * grad
        
        return -self.eta * self.momentum
