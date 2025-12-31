import torch
import torch.nn as nn
from layers.cell_processing import CellProcessing

class SimpleTabularModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        self.d_model = d_model
        
        # Cell processing
        self.cell_processing = CellProcessing(
            d_model=d_model,
            vocab_size=config.get('vocab_size', 50000),
            n_types=config.get('n_types', 10),
            max_cols=config.get('max_cols', 64)
        )
        
        # Simple MLP layers
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        
        # Heads
        self.classification_head = nn.Linear(d_model, config['n_classes'])
        self.regression_head = nn.Linear(d_model, 1)
        
        self.model_name = "SimpleTabularModel"
        self.version = "1.0"
        
    def forward(self, values, types=None, task='classification', continuous=True, **kwargs):
        # Cell processing
        x = self.cell_processing(values, types, continuous=continuous)
        
        # MLP
        x = self.layers(x)
        
        # Pool
        x = x.mean(dim=1)
        
        # Head
        if task == 'classification':
            out = self.classification_head(x)
        else:
            out = self.regression_head(x)
            
        return {
            'base_output': out,
            'domain_output': None,
            'memory_state': None,
            'midas_loss': torch.tensor(0.0, device=values.device)
        }
    
    def get_info(self):
        return {
            'name': self.model_name,
            'version': self.version,
            'params': sum(p.numel() for p in self.parameters()),
            'd_model': self.d_model,
        }
