import torch
import torch.nn as nn
from layers.cell_processing import CellProcessing
from layers.schema_processing import SchemaProcessing
from layers.local_reasoning import LocalReasoningLayer
from layers.global_reasoning import GlobalReasoningLayer
from adapters.schema_adapter import DomainSchemaAdapter
from adapters.knowledge_injection import DomainKnowledgeInjection
from adapters.domain_heads import DomainSpecificHeads
from layers.midas import MIDAS

class TabularFoundationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        self.d_model = d_model
        n_features = config.get('n_features', 10)
        
        # MIDAS - Missing Data Imputation
        self.midas = MIDAS(d_input=n_features, d_hidden=128)
        
        # Cell Processing
        self.cell_processing = CellProcessing(
            d_model=d_model,
            vocab_size=config['vocab_size'],
            n_types=config['n_types'],
            max_cols=config.get('max_cols', 64)
        )
        
        # MIRAS: Local Reasoning
        self.local_reasoning = LocalReasoningLayer(
            d_model=d_model,
            n_heads=config['n_heads']
        )
        
        # MIRAS: Global Reasoning
        self.global_reasoning = GlobalReasoningLayer(
            d_model=d_model,
            n_heads=config['n_heads'],
            n_latents=config['n_latents'],
            sector=config.get('sector', 'default')
        )
        
        # Heads
        self.classification_head = nn.Linear(d_model, config['n_classes'])
        self.regression_head = nn.Linear(d_model, 1)
        
        # Backward compat
        self.schema_processing = SchemaProcessing(d_model=d_model, n_heads=config['n_heads'], n_layers=2)
        self.schema_adapter = DomainSchemaAdapter(d_model=d_model, n_domains=config['n_domains'])
        self.knowledge_injection = DomainKnowledgeInjection(d_model=d_model, n_domains=config['n_domains'])
        self.schema_cell_fusion = nn.Linear(d_model * 2, d_model)
        self.domain_heads = DomainSpecificHeads(d_model=d_model, n_domains=config['n_domains'])
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, values, types=None, mask=None, task='classification', continuous=True, **kwargs):
        
        # 1. MIDAS - Handle missing data
        midas_loss = torch.tensor(0.0, device=values.device)
        if mask is not None:
            values, midas_loss = self.midas(values, mask)
        
        # 2. Cell processing
        x = self.cell_processing(values, types, continuous=continuous)
        
        # 3. Local reasoning
        x = self.local_reasoning(x)
        
        # 4. Global reasoning
        x, memory_state = self.global_reasoning(x)
        
        # 5. Pool
        pooled = x.mean(dim=1)
        
        # 6. Output
        if task == 'classification':
            base_output = self.classification_head(pooled)
        else:
            base_output = self.regression_head(pooled)
            
        return {
            'base_output': base_output,
            'domain_output': None,
            'memory_state': memory_state,
            'midas_loss': midas_loss
        }


class SchemalabsBaseModel12v1(TabularFoundationModel):
    """V1.2 - With MIRAS + MIDAS + Online Learning"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_name = "SchemalabsBaseModel12"
        self.version = "1.2"
        self.online_learning_enabled = False
        
        # Online learning (EWC)
        self.ewc_lambda = 1000
        self.fisher_info = {}
        self.optimal_params = {}
        
    def enable_online_learning(self):
        self.online_learning_enabled = True
        self._store_optimal_params()
        
    def disable_online_learning(self):
        self.online_learning_enabled = False
        
    def _store_optimal_params(self):
        for n, p in self.named_parameters():
            self.optimal_params[n] = p.clone().detach()
            
    def compute_fisher(self, dataloader, criterion):
        """Compute Fisher information for EWC"""
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.eval()
        
        for batch in dataloader:
            x, y = batch
            self.zero_grad()
            out = self(values=x, continuous=True)
            loss = criterion(out['base_output'], y)
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.pow(2)
                    
        for n in self.fisher_info:
            self.fisher_info[n] /= len(dataloader)
            
    def ewc_loss(self):
        """Elastic Weight Consolidation loss"""
        if not self.fisher_info:
            return torch.tensor(0.0)
            
        loss = 0
        for n, p in self.named_parameters():
            if n in self.fisher_info:
                loss += (self.fisher_info[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.ewc_lambda * loss
        
    def get_info(self):
        return {
            'name': self.model_name,
            'version': self.version,
            'params': sum(p.numel() for p in self.parameters()),
            'd_model': self.d_model,
            'features': {
                'midas': True,
                'miras': True,
                'online_learning': self.online_learning_enabled
            }
        }
