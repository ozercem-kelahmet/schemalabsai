import torch
import torch.nn as nn

class TableClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, global_latents):
        return self.head(global_latents)


class CellImputationHead(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_features)
        )
        
    def forward(self, cell_embeddings):
        return self.head(cell_embeddings)


class RowPredictionHead(nn.Module):
    def __init__(self, d_model, n_outputs=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_outputs)
        )
        
    def forward(self, row_embeddings):
        return self.head(row_embeddings)


class CellHead(nn.Module):
    """Masked Cell Modeling (MCM) - BERT MLM gibi self-supervised learning"""
    def __init__(self, d_model, n_features):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # Her hücre için değer tahmini
        )
    
    def forward(self, cell_embeddings):
        # cell_embeddings: (batch, n_cols, d_model)
        return self.head(cell_embeddings).squeeze(-1)  # (batch, n_cols)


class TaskSpecificHeads(nn.Module):
    def __init__(self, d_model, n_classes, n_features):
        super().__init__()
        self.classification = TableClassificationHead(d_model, n_classes)
        self.imputation = CellImputationHead(d_model, n_features)
        self.row_prediction = RowPredictionHead(d_model)
        
    def forward(self, global_latents, cell_embeddings=None, task='classification'):
        if task == 'classification':
            return self.classification(global_latents)
        elif task == 'imputation' and cell_embeddings is not None:
            return self.imputation(cell_embeddings)
        elif task == 'row_prediction':
            return self.row_prediction(global_latents)
        return self.classification(global_latents)
