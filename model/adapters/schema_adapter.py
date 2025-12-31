import torch
import torch.nn as nn

class DomainSchemaAdapter(nn.Module):
    def __init__(self, d_model, n_domains=20):
        super().__init__()
        self.domain_rules = nn.Embedding(n_domains, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, schema_embeddings, domain_id):
        domain_embed = self.domain_rules(domain_id)
        domain_embed = domain_embed.unsqueeze(1).expand(-1, schema_embeddings.shape[1], -1)
        
        fused = torch.cat([schema_embeddings, domain_embed], dim=-1)
        output = self.fusion(fused)
        
        return self.norm(output)
