import torch
import torch.nn as nn

class DomainSpecificHeads(nn.Module):
    def __init__(self, d_model, n_domains=1000, n_classes_per_domain=10):
        super().__init__()
        self.d_model = d_model
        self.default_head = nn.Linear(d_model, n_classes_per_domain)
        self.heads = nn.ModuleDict()
        
    def forward(self, x, domain_name='default'):
        if domain_name in self.heads:
            return self.heads[domain_name](x)
        return self.default_head(x)
    
    def add_head(self, domain_name, n_classes):
        self.heads[domain_name] = nn.Linear(self.d_model, n_classes)
