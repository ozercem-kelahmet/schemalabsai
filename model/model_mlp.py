import torch
import torch.nn as nn
from layers.midas import MIDAS

class SchemalabsMLP(nn.Module):
    """MLP-based model with MIDAS + Self-Learning (EWC)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_features = config.get('n_features', 10)
        n_classes = config['n_classes']
        d_hidden = config.get('d_hidden', 1024)
        
        # MIDAS - Missing Data Imputation
        self.midas = MIDAS(d_input=n_features, d_hidden=128)
        
        # MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Heads
        self.classification_head = nn.Linear(d_hidden // 2, n_classes)
        self.regression_head = nn.Linear(d_hidden // 2, 1)
        
        # Self-Learning (EWC)
        self.online_learning_enabled = False
        self.ewc_lambda = 5000  # Artırıldı
        self.fisher_info = {}
        self.optimal_params = {}
        
        self.model_name = "SchemalabsMLP"
        self.version = "1.0"
        
    def forward(self, values, mask=None, task='classification', **kwargs):
        midas_loss = torch.tensor(0.0, device=values.device)
        if mask is not None:
            values, midas_loss = self.midas(values, mask)
        
        x = self.backbone(values)
        
        if task == 'classification':
            base_output = self.classification_head(x)
        else:
            base_output = self.regression_head(x)
            
        return {
            'base_output': base_output,
            'memory_state': None,
            'midas_loss': midas_loss
        }
    
    def enable_online_learning(self):
        self.online_learning_enabled = True
        self._store_optimal_params()
        
    def disable_online_learning(self):
        self.online_learning_enabled = False
        
    def _store_optimal_params(self):
        for n, p in self.named_parameters():
            self.optimal_params[n] = p.data.clone()
            
    def compute_fisher(self, dataloader, criterion, samples=1000):
        """Compute Fisher information for EWC"""
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.train()
        
        count = 0
        for batch in dataloader:
            if count >= samples:
                break
            x, y = batch
            self.zero_grad()
            out = self(values=x)
            loss = criterion(out['base_output'], y)
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.data.pow(2)
            count += len(x)
                    
        for n in self.fisher_info:
            self.fisher_info[n] /= count
            
        print(f"Fisher computed on {count} samples")
            
    def ewc_loss(self):
        """Elastic Weight Consolidation loss"""
        if not self.fisher_info or not self.optimal_params:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for n, p in self.named_parameters():
            if n in self.fisher_info and n in self.optimal_params:
                loss += (self.fisher_info[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.ewc_lambda * loss
    
    def get_info(self):
        return {
            'name': self.model_name,
            'version': self.version,
            'params': sum(p.numel() for p in self.parameters()),
            'features': {
                'midas': True,
                'self_learning': self.online_learning_enabled
            }
        }
