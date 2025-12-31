import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class EWC:
    """
    Elastic Weight Consolidation
    Eski bilgiyi koruyarak yeni şeyler öğrenir.
    Catastrophic Forgetting'i önler.
    """
    
    def __init__(self, model, importance=1000):
        self.model = model
        self.importance = importance  # Lambda - ne kadar sıkı koruma
        self.fisher = {}              # Her parametre için önem skoru
        self.old_params = {}          # Eski parametre değerleri
        self.initialized = False
        
    def compute_fisher(self, dataloader, device='cpu'):
        """
        Fisher Information Matrix hesapla.
        Her parametre ne kadar önemli?
        """
        self.model.eval()
        
        # Fisher'ı sıfırla
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
        
        n_samples = 0
        
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
            else:
                x = batch[0].to(device)
                y = None
            
            self.model.zero_grad()
            
            out = self.model(values=x, continuous=True, task='classification')
            
            # Log-likelihood
            if y is not None:
                log_probs = torch.log_softmax(out['base_output'], dim=-1)
                loss = -log_probs.gather(1, y.unsqueeze(1)).mean()
            else:
                # Pseudo-label kullan
                probs = torch.softmax(out['base_output'], dim=-1)
                log_probs = torch.log_softmax(out['base_output'], dim=-1)
                loss = -(probs * log_probs).sum(dim=-1).mean()
            
            loss.backward()
            
            # Fisher = gradient^2
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data.clone() ** 2
            
            n_samples += x.size(0)
        
        # Normalize
        for name in self.fisher:
            self.fisher[name] /= n_samples
        
        # Eski parametreleri kaydet
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        self.initialized = True
        print(f"[EWC] Fisher computed on {n_samples} samples")
        
    def penalty(self):
        """
        EWC penalty hesapla.
        Önemli parametreler değiştiyse ceza ver.
        """
        if not self.initialized:
            return torch.tensor(0.0)
        
        loss = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher:
                loss += (self.fisher[name] * (param - self.old_params[name]) ** 2).sum()
        
        return self.importance * loss
    
    def update_fisher(self, dataloader, device='cpu', decay=0.9):
        """
        Fisher'ı güncelle (yeni veriyle).
        Decay ile eski bilgiyi koru.
        """
        old_fisher = {k: v.clone() for k, v in self.fisher.items()}
        
        self.compute_fisher(dataloader, device)
        
        # Eski ve yeni Fisher'ı birleştir
        if old_fisher:
            for name in self.fisher:
                if name in old_fisher:
                    self.fisher[name] = decay * old_fisher[name] + (1 - decay) * self.fisher[name]
        
        print(f"[EWC] Fisher updated with decay={decay}")
    
    def save(self, path):
        """EWC state kaydet"""
        torch.save({
            'fisher': self.fisher,
            'old_params': self.old_params,
            'importance': self.importance,
            'initialized': self.initialized
        }, path)
        print(f"[EWC] Saved to {path}")
    
    def load(self, path):
        """EWC state yükle"""
        state = torch.load(path)
        self.fisher = state['fisher']
        self.old_params = state['old_params']
        self.importance = state['importance']
        self.initialized = state['initialized']
        print(f"[EWC] Loaded from {path}")


class OnlineEWC(EWC):
    """
    Online EWC - Sürekli güncellenen versiyon.
    Her güncelleme sonrası Fisher'ı birleştirir.
    """
    
    def __init__(self, model, importance=1000, gamma=0.95):
        super().__init__(model, importance)
        self.gamma = gamma  # Eski Fisher'ın ağırlığı
        self.task_count = 0
        
    def register_new_task(self, dataloader, device='cpu'):
        """
        Yeni task/veri geldiğinde çağır.
        Fisher'ı günceller, eskiyi korur.
        """
        if self.task_count == 0:
            self.compute_fisher(dataloader, device)
        else:
            # Eski Fisher'ı sakla
            old_fisher = {k: v.clone() for k, v in self.fisher.items()}
            
            # Yeni Fisher hesapla
            self.compute_fisher(dataloader, device)
            
            # Birleştir: gamma * old + (1-gamma) * new
            for name in self.fisher:
                if name in old_fisher:
                    self.fisher[name] = self.gamma * old_fisher[name] + self.fisher[name]
        
        self.task_count += 1
        print(f"[OnlineEWC] Task {self.task_count} registered")
