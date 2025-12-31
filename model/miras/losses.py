import torch
import torch.nn as nn

class MultiObjectiveLoss(nn.Module):
    def __init__(self, sector='default'):
        super().__init__()
        self.sector = sector
        self.loss_fn = nn.CrossEntropyLoss()  # Classification için
        
    def forward(self, pred, target):
        return self.loss_fn(pred, target)
