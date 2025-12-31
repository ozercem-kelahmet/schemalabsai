import torch
import torch.nn as nn
from layers.cell_processing import CellProcessing
from layers.schema_processing import SchemaProcessing
from layers.local_reasoning import LocalReasoningLayer
from layers.global_reasoning import GlobalReasoningLayer
from layers.midas import MIDAS
from adapters.schema_adapter import DomainSchemaAdapter
from adapters.knowledge_injection import DomainKnowledgeInjection
from adapters.domain_heads import DomainSpecificHeads

class SchemalabsBaseModel12v1(nn.Module):
    """
    SchemalabsAI Base Model v1.2
    
    Features:
    - MIDAS: Missing Data Imputation
    - MIRAS: Memory System (12 features)
    - Persistent Memory
    - Online Learning Ready
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = "SchemalabsBaseModel12v1"
        self.version = "1.2"
        
        d_model = config['d_model']
        self.d_model = d_model
        
        # MIDAS - Missing Data Imputation
        self.midas = MIDAS(d_input=config.get('n_features', 10), d_hidden=128)
        
        self.cell_processing = CellProcessing(
            d_model=d_model,
            vocab_size=config['vocab_size'],
            n_types=config['n_types'],
            max_cols=config.get('max_cols', 64)
        )
        self.schema_processing = SchemaProcessing(
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config.get('schema_layers', 2)
        )
        self.schema_adapter = DomainSchemaAdapter(
            d_model=d_model,
            n_domains=config['n_domains']
        )
        self.local_reasoning = LocalReasoningLayer(
            d_model=d_model,
            n_heads=config['n_heads']
        )
        self.global_reasoning = GlobalReasoningLayer(
            d_model=d_model,
            n_heads=config['n_heads'],
            n_latents=config['n_latents'],
            sector=config.get('sector', 'default')
        )
        self.knowledge_injection = DomainKnowledgeInjection(
            d_model=d_model,
            n_domains=config['n_domains']
        )
        self.schema_cell_fusion = nn.Linear(d_model * 2, d_model)
        
        self.classification_head = nn.Linear(d_model, config['n_classes'])
        self.regression_head = nn.Linear(d_model, 1)
        self.domain_heads = DomainSpecificHeads(
            d_model=d_model,
            n_domains=config['n_domains']
        )
        self.gate = nn.Linear(d_model, d_model)
        self.memory_alpha = 0.7
        
        # Online learning flag
        self.online_learning_enabled = False
        
    def forward(self, values, types=None, schema_info=None, domain_id=None,
                domain_name='default', task='classification', continuous=False,
                prev_memory=None, missing_mask=None):
        batch_size = values.shape[0]
        device = values.device
        
        # MIDAS - Eksik veri doldurma
        values, midas_loss = self.midas(values, missing_mask)
        
        cell_features = self.cell_processing(values, types, continuous=continuous)
        
        if schema_info is None:
            schema_info = cell_features.detach()
        if domain_id is None:
            domain_id = torch.zeros(batch_size, dtype=torch.long, device=device)
            
        schema_features = self.schema_processing(schema_info)
        schema_features = self.schema_adapter(schema_features, domain_id)
        
        combined = torch.cat([cell_features, schema_features], dim=-1)
        fused_features = self.schema_cell_fusion(combined)
        
        local_features = self.local_reasoning(fused_features)
        global_features, memory_state = self.global_reasoning(local_features)
        
        # Persistent Memory - Alpha blending (MIRAS)
        if prev_memory is not None:
            memory_state = self.memory_alpha * prev_memory + (1 - self.memory_alpha) * memory_state
        
        global_features = self.knowledge_injection(global_features, domain_id)
        
        gate = torch.sigmoid(self.gate(global_features))
        gated_output = global_features * gate
        
        pooled = gated_output.mean(dim=1)
        
        if task == 'classification':
            base_output = self.classification_head(pooled)
        elif task == 'regression':
            base_output = self.regression_head(pooled)
        else:
            base_output = None
            
        domain_output = self.domain_heads(pooled, domain_name)
        
        return {
            'base_output': base_output,
            'domain_output': domain_output,
            'memory_state': memory_state,
            'midas_loss': midas_loss
        }
    
    def enable_online_learning(self):
        """Online learning'i aktif et"""
        self.online_learning_enabled = True
        
    def disable_online_learning(self):
        """Online learning'i kapat"""
        self.online_learning_enabled = False
    
    def get_info(self):
        """Model bilgilerini döndür"""
        return {
            'name': self.model_name,
            'version': self.version,
            'params': sum(p.numel() for p in self.parameters()),
            'd_model': self.d_model,
            'features': {
                'midas': True,
                'miras': True,
                'persistent_memory': True,
                'online_learning': self.online_learning_enabled
            }
        }
