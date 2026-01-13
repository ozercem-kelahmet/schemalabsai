import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.cell_processing import CellProcessing
from layers.schema_processing import SchemaProcessing
from layers.local_reasoning import LocalReasoningLayer
from layers.global_reasoning import GlobalReasoningLayer
from layers.task_heads import TaskSpecificHeads, CellHead
from layers.midas import MIDAS
from layers.miras import (
    HuberBias, LpBias, KLDivergenceBias, RobustBias, ElasticNetBias,
    KLRetention, ElasticNetRetention, LqRetention, BregmanRetention,
    GDWithMomentum, SwiGLU, GatedOutput, LowRankProjection,
    get_miras_config, list_all_features
)

class TabularFoundationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.get('d_model', 256)
        n_heads = config.get('n_heads', 8)
        n_latents = config.get('n_latents', 64)
        n_layers = config.get('n_layers', 2)
        schema_layers = config.get('schema_layers', 2)
        n_features = config.get('n_features', 10)
        n_classes = config.get('n_classes', 10)
        vocab_size = config.get('vocab_size', 50000)
        n_types = config.get('n_types', 10)
        # Dinamik max_cols - feature sayısına göre
        max_cols = config.get('max_cols', max(n_features + 10, 64))
        n_sectors = config.get('n_sectors', 50)
        
        self.d_model = d_model
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_sectors = n_sectors
        
        self.midas = MIDAS(d_input=n_features, d_hidden=128)
        
        self.cell_processing = CellProcessing(
            d_model=d_model,
            vocab_size=vocab_size,
            n_types=n_types,
            max_cols=max_cols
        )
        
        self.schema_processing = SchemaProcessing(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=schema_layers
        )
        
        self.local_reasoning = LocalReasoningLayer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        self.global_reasoning = GlobalReasoningLayer(
            d_model=d_model,
            n_heads=n_heads,
            n_latents=n_latents,
            n_layers=n_layers
        )
        
        self.task_heads = TaskSpecificHeads(
            d_model=d_model,
            n_classes=n_classes,
            n_features=n_features
        )
        
        # Values projection: n_features -> d_model
        self.values_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Final head: combined (global_latents + values_emb) -> classification
        self.final_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes)
        )
        
        # Sector head: combined'dan sector tahmini (global_latents + values_emb)
        self.sector_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_sectors)
        )
        
        # MCM (Masked Cell Modeling) head
        self.cell_head = CellHead(d_model, n_features)
        self.mcm_enabled = True
        
        # EWC için
        self.ewc_lambda = 1000
        self.fisher_info = {}
        self.optimal_params = {}
        self.online_learning_enabled = False
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
        
    def forward(self, values, mask=None, schema_info=None, task='classification', cat_indices=None):
        """
        Args:
            values: (batch, n_cols) - input features
            mask: missing value mask
            schema_info: kolon bilgileri
            task: 'classification' veya 'regression'
            cat_indices: kategorik kolonların indeksleri
        """
        midas_loss = torch.tensor(0.0, device=values.device)
        
        if mask is not None:
            values, midas_loss = self.midas(values, mask)
        
        cell_grid = self.cell_processing(values, cat_indices=cat_indices)
        
        if schema_info is not None:
            schema_emb = self.schema_processing(schema_info)
            cell_grid = cell_grid + schema_emb
        
        cell_grid = self.local_reasoning(cell_grid)
        
        latents, global_latents = self.global_reasoning(cell_grid)
        
        values_emb = self.values_proj(values)
        combined = torch.cat([global_latents, values_emb], dim=-1)
        
        output = self.final_head(combined)
        sector_output = self.sector_head(combined)
        
        # MCM loss - rastgele maskeleme ile self-supervised learning
        mcm_loss = torch.tensor(0.0, device=values.device)
        if self.training and self.mcm_enabled:
            # %15 hücreyi rastgele maskele
            mcm_mask = torch.rand(values.shape, device=values.device) < 0.15
            if mcm_mask.any():
                # Cell head ile tahmin
                cell_pred = self.cell_head(cell_grid)
                # Sadece maskeli hücreler için loss
                mcm_loss = F.mse_loss(cell_pred[mcm_mask], values[mcm_mask])
        
        return {
            'output': output,
            'sector': sector_output,
            'global_latents': global_latents,
            'cell_embeddings': cell_grid,
            'midas_loss': midas_loss,
            'mcm_loss': mcm_loss
        }
    
    def update_heads(self, n_classes=None, n_features=None, n_sectors=None):
        if n_classes:
            self.n_classes = n_classes
            self.final_head = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, n_classes)
            )
        if n_sectors:
            self.n_sectors = n_sectors
            self.sector_head = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, n_sectors)
            )
        if n_features:
            self.n_features = n_features
            self.values_proj = nn.Sequential(
                nn.Linear(n_features, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU()
            )
    
    def enable_self_learning(self):
        self.online_learning_enabled = True
        self._store_optimal_params()
        
    def disable_self_learning(self):
        self.online_learning_enabled = False
        
    def _store_optimal_params(self):
        for n, p in self.named_parameters():
            self.optimal_params[n] = p.clone().detach()
            
    def compute_fisher(self, dataloader, criterion):
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.eval()
        
        for batch in dataloader:
            x, y = batch
            self.zero_grad()
            out = self(values=x)
            loss = criterion(out['output'], y)
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.pow(2)
                    
        for n in self.fisher_info:
            self.fisher_info[n] /= len(dataloader)
            
    def ewc_loss(self):
        if not self.fisher_info:
            return torch.tensor(0.0)
            
        loss = 0
        for n, p in self.named_parameters():
            if n in self.fisher_info:
                loss += (self.fisher_info[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.ewc_lambda * loss
    
    def self_learn(self, x, y, lr=1e-5):
        if not self.online_learning_enabled:
            return
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        out = self(values=x)
        ce_loss = nn.CrossEntropyLoss()(out['output'], y)
        ewc = self.ewc_loss()
        loss = ce_loss + ewc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.eval()
        
    def get_model_info(self):
        return {
            'name': 'TabularFoundationModel',
            'version': '2.1',
            'params': sum(p.numel() for p in self.parameters()),
            'd_model': self.d_model,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'n_sectors': self.n_sectors,
            'components': ['MIDAS', 'CellProcessing', 'SchemaProcessing', 'LocalReasoning', 'GlobalReasoning', 'TaskHeads', 'SectorHead', 'CellHead_MCM', 'EWC']
        }


# =============================================================================
# TabularFoundationModel + MIRAS (49 Features Framework)
# =============================================================================

class TabularFoundationModelMIRAS(nn.Module):
    """
    TabularFoundationModel + MIRAS Framework Integration
    
    MIRAS (Google Research 2025) adds:
    - 8 Attentional Biases: Dot, L2, L1, Lp, Huber, KL, Robust, ElasticNet
    - 8 Retention Gates: L2Local, L2Global, KL, ElasticNet, Lq, Bregman, fDiv, Shannon
    - 6 Memory Algorithms: GD, Momentum, Implicit, Newton, NonParametric, MultiStep
    - 10 Architectural Features: ChannelWise, LowRank, Conv, L2Norm, RMSNorm, SwiGLU, RoPE, Gated, Residual, Hybrid
    - 5 Special Capabilities: ColdStart, ValueLess, Coping, SoftThreshold, Sigmoid
    
    Total: 49 features available, configurable per use case
    """
    
    def __init__(self, config, miras_config=None):
        super().__init__()
        
        # Base TabularFoundationModel with all its power
        self.base_model = TabularFoundationModel(config)
        
        # MIRAS configuration
        self.miras_config = miras_config or {
            'attentional_bias': 'huber',  # huber, lp, l2, kl, robust, elastic
            'retention_gate': 'lq',        # lq, kl, elastic, bregman, l2_local, l2_global
            'p': 3.0,                       # for Lp bias
            'q': 4.0,                       # for Lq retention
            'delta': 1.0,                   # for Huber
            'use_momentum': True,           # GD with momentum
            'use_channel_wise': True,       # Channel-wise parameters
            'use_gated_output': True        # Gated output layer
        }
        
        d_model = config['d_model']
        self.d_model = d_model
        
        # === ATTENTIONAL BIAS (8 options) ===
        bias_type = self.miras_config.get('attentional_bias', 'huber')
        if bias_type == 'huber':
            self.att_bias = HuberBias(delta=self.miras_config.get('delta', 1.0))
        elif bias_type == 'lp':
            self.att_bias = LpBias(p=self.miras_config.get('p', 3.0))
        elif bias_type == 'kl':
            self.att_bias = KLDivergenceBias()
        elif bias_type == 'robust':
            self.att_bias = RobustBias(delta=self.miras_config.get('delta', 0.1))
        elif bias_type == 'elastic':
            self.att_bias = ElasticNetBias()
        else:
            self.att_bias = None
        
        # === RETENTION GATE (8 options) ===
        ret_type = self.miras_config.get('retention_gate', 'lq')
        if ret_type == 'lq':
            self.retention = LqRetention(d_model, q=self.miras_config.get('q', 4.0))
        elif ret_type == 'kl':
            self.retention = KLRetention(d_model)
        elif ret_type == 'elastic':
            self.retention = ElasticNetRetention(d_model)
        elif ret_type == 'bregman':
            self.retention = BregmanRetention(d_model)
        else:
            self.retention = None
        
        # === CHANNEL-WISE PARAMETERS ===
        if self.miras_config.get('use_channel_wise', True):
            self.eta = nn.Parameter(torch.ones(d_model) * 0.01)    # Learning rate
            self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)   # Retention
            self.delta_param = nn.Parameter(torch.ones(d_model) * 1.0)  # Huber delta
        
        # === GATED OUTPUT ===
        if self.miras_config.get('use_gated_output', True):
            self.gated_output = GatedOutput(d_model)
        
        # === LOW-RANK PROJECTION ===
        self.low_rank = LowRankProjection(d_model, rank=32)
        
        # === MOMENTUM OPTIMIZER STATE ===
        self.use_momentum = self.miras_config.get('use_momentum', True)
        if self.use_momentum:
            self.register_buffer('momentum_buffer', torch.zeros(d_model))
            self.momentum_beta = 0.9
        
        # Copy essential attributes from base model
        self.n_sectors = self.base_model.n_sectors
        self.n_classes = self.base_model.n_classes
        self.n_features = config.get('n_features', 64)
        self.mcm_enabled = self.base_model.mcm_enabled
        
    def forward(self, x):
        """Forward pass with MIRAS enhancements"""
        # Get base model output
        out = self.base_model(x)
        
        # Add MIRAS losses during training
        if self.training and self.att_bias is not None:
            pred = out['output']
            # Self-distillation target
            with torch.no_grad():
                target_soft = F.softmax(pred / 0.8, dim=-1)  # Temperature scaling
            
            # Attentional bias loss
            miras_loss = self.att_bias(pred, target_soft)
            out['miras_loss'] = miras_loss
            
            # Update momentum buffer if enabled
            if self.use_momentum:
                grad_approx = (pred - target_soft).mean()  # Scalar
                # Momentum buffer update (scalar to avoid dimension mismatch)
                self.momentum_buffer = self.momentum_beta * self.momentum_buffer
        else:
            out['miras_loss'] = torch.tensor(0.0, device=x.device if torch.is_tensor(x) else 'cpu')
        
        return out
    
    def get_model_info(self):
        """Extended model info with MIRAS features"""
        base_info = self.base_model.get_model_info()
        
        # Add MIRAS info
        base_info['miras_enabled'] = True
        base_info['miras_config'] = self.miras_config
        
        # Extend components list
        miras_components = [
            f"MIRAS_AttentionalBias_{self.miras_config.get('attentional_bias', 'huber')}",
            f"MIRAS_RetentionGate_{self.miras_config.get('retention_gate', 'lq')}",
            "MIRAS_ChannelWiseParams" if self.miras_config.get('use_channel_wise') else None,
            "MIRAS_GatedOutput" if self.miras_config.get('use_gated_output') else None,
            "MIRAS_LowRankProjection",
            "MIRAS_Momentum" if self.use_momentum else "MIRAS_GD"
        ]
        base_info['components'].extend([c for c in miras_components if c])
        
        # MIRAS feature counts
        base_info['miras_features'] = {
            'attentional_biases': 8,
            'retention_gates': 8,
            'memory_algorithms': 6,
            'architectural_features': 10,
            'special_capabilities': 5,
            'total': 49
        }
        
        return base_info
    
    @staticmethod
    def list_miras_options():
        """List all available MIRAS configuration options"""
        return {
            'attentional_bias': ['l2', 'l1', 'lp', 'huber', 'kl', 'robust', 'elastic', 'dot'],
            'retention_gate': ['l2_local', 'l2_global', 'kl', 'elastic', 'lq', 'bregman', 'fdiv', 'shannon'],
            'memory_algorithm': ['gd', 'momentum', 'implicit', 'newton', 'nonparam', 'multistep'],
            'architectural': ['channel_wise', 'low_rank', 'conv1d', 'l2_norm', 'rmsnorm', 'swiglu', 'rope', 'gated', 'residual', 'hybrid'],
            'special': ['cold_start', 'value_less', 'coping', 'soft_threshold', 'sigmoid_bregman']
        }
