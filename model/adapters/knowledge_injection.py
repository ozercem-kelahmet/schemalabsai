import torch
import torch.nn as nn

class DomainKnowledgeInjection(nn.Module):
    def __init__(self, d_model, n_domains=20):
        super().__init__()
        self.domain_context = nn.Embedding(n_domains, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, global_features, domain_id):
        domain_embed = self.domain_context(domain_id)
        domain_embed = domain_embed.unsqueeze(1)
        
        attended, _ = self.cross_attn(global_features, domain_embed, domain_embed)
        
        output = global_features + attended
        
        return self.norm(output)
