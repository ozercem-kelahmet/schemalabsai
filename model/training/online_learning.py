import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from collections import deque
import time
import json
import sys
sys.path.append('..')

from layers.ewc import OnlineEWC


class PseudoLabeler:
    """
    Güvenilir pseudo-label üretir.
    Sadece yüksek confidence tahminleri kullanır.
    """
    
    def __init__(self, confidence_threshold=0.90, agreement_threshold=0.95):
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        
    def get_pseudo_labels(self, model, x, use_ensemble=True):
        """
        Güvenilir pseudo-label üret.
        
        Returns:
            labels: Pseudo-labels (güvenilir olanlar)
            mask: Hangileri güvenilir (True = kullan)
            confidences: Her örneğin confidence'ı
        """
        model.eval()
        
        with torch.no_grad():
            if use_ensemble:
                # Dropout ensemble (5 forward pass)
                model.train()  # Dropout aktif
                preds_list = []
                for _ in range(5):
                    out = model(values=x, continuous=True, task='classification')
                    probs = torch.softmax(out['base_output'], dim=-1)
                    preds_list.append(probs)
                model.eval()
                
                # Ortalama prediction
                avg_probs = torch.stack(preds_list).mean(dim=0)
                
                # Agreement: Tüm modeller aynı şeyi mi diyor?
                all_preds = torch.stack([p.argmax(dim=-1) for p in preds_list])
                mode_preds = avg_probs.argmax(dim=-1)
                agreement = (all_preds == mode_preds.unsqueeze(0)).float().mean(dim=0)
                
                confidences = avg_probs.max(dim=-1).values
                labels = avg_probs.argmax(dim=-1)
                
                # Güvenilir: Yüksek confidence + Yüksek agreement
                mask = (confidences >= self.confidence_threshold) & \
                       (agreement >= self.agreement_threshold)
            else:
                out = model(values=x, continuous=True, task='classification')
                probs = torch.softmax(out['base_output'], dim=-1)
                confidences = probs.max(dim=-1).values
                labels = probs.argmax(dim=-1)
                
                mask = confidences >= self.confidence_threshold
        
        return labels, mask, confidences
    
    def filter_reliable(self, x, labels, mask):
        """Sadece güvenilir örnekleri döndür"""
        reliable_x = x[mask]
        reliable_labels = labels[mask]
        return reliable_x, reliable_labels


class UncertaintyBuffer:
    """
    Düşük confidence örnekleri saklar.
    Belirli bir kapasitede tutar (FIFO).
    """
    
    def __init__(self, max_size=10000, uncertainty_threshold=0.7):
        self.buffer = deque(maxlen=max_size)
        self.uncertainty_threshold = uncertainty_threshold
        
    def add(self, x, confidence, prediction):
        """
        Düşük confidence örneği ekle.
        """
        if confidence < self.uncertainty_threshold:
            self.buffer.append({
                'input': x.cpu().numpy(),
                'confidence': confidence.item() if torch.is_tensor(confidence) else confidence,
                'prediction': prediction.item() if torch.is_tensor(prediction) else prediction,
                'timestamp': time.time()
            })
    
    def add_batch(self, x, confidences, predictions):
        """Batch olarak ekle"""
        for i in range(len(x)):
            self.add(x[i], confidences[i], predictions[i])
    
    def get_samples(self, n=None):
        """Buffer'dan örnek al"""
        if n is None:
            n = len(self.buffer)
        
        samples = list(self.buffer)[:n]
        
        if not samples:
            return None, None
        
        x = torch.FloatTensor(np.array([s['input'] for s in samples]))
        preds = torch.LongTensor([s['prediction'] for s in samples])
        
        return x, preds
    
    def clear(self):
        """Buffer'ı temizle"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class UpdateScheduler:
    """
    Ne zaman model güncelleneceğine karar verir.
    Kalite kontrolü yapar.
    """
    
    def __init__(self, 
                 min_samples=500,
                 min_avg_confidence=0.85,
                 min_reliable_ratio=0.3,
                 max_wait_seconds=7*24*3600,  # 1 hafta
                 validation_threshold=0.95):
        self.min_samples = min_samples
        self.min_avg_confidence = min_avg_confidence
        self.min_reliable_ratio = min_reliable_ratio
        self.max_wait_seconds = max_wait_seconds
        self.validation_threshold = validation_threshold
        self.last_update = time.time()
        
    def should_update(self, buffer, reliable_count, total_count, val_accuracy=None):
        """
        Güncelleme yapılmalı mı?
        
        Args:
            buffer: UncertaintyBuffer
            reliable_count: Güvenilir pseudo-label sayısı
            total_count: Toplam örnek sayısı
            val_accuracy: Validation accuracy (opsiyonel)
        
        Returns:
            should_update: bool
            reason: str
        """
        # 1. Yeterli örnek var mı?
        if len(buffer) < self.min_samples:
            return False, f"Not enough samples: {len(buffer)}/{self.min_samples}"
        
        # 2. Güvenilir oran yeterli mi?
        if total_count > 0:
            reliable_ratio = reliable_count / total_count
            if reliable_ratio < self.min_reliable_ratio:
                return False, f"Low reliable ratio: {reliable_ratio:.2f}/{self.min_reliable_ratio}"
        
        # 3. Max bekleme süresi aşıldı mı?
        time_since_update = time.time() - self.last_update
        if time_since_update > self.max_wait_seconds:
            return True, f"Max wait time exceeded: {time_since_update/3600:.1f}h"
        
        # 4. Validation accuracy kontrolü
        if val_accuracy is not None and val_accuracy < self.validation_threshold:
            return False, f"Validation accuracy too low: {val_accuracy:.2f}/{self.validation_threshold}"
        
        return True, "Ready for update"
    
    def mark_updated(self):
        """Güncelleme yapıldığını işaretle"""
        self.last_update = time.time()


class OnlineLearner:
    """
    Ana online learning sistemi.
    Self-learning + EWC entegrasyonu.
    """
    
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # Bileşenler
        self.ewc = OnlineEWC(model, importance=1000, gamma=0.95)
        self.pseudo_labeler = PseudoLabeler(confidence_threshold=0.90)
        self.buffer = UncertaintyBuffer(max_size=10000, uncertainty_threshold=0.7)
        self.scheduler = UpdateScheduler(min_samples=500)
        
        # İstatistikler
        self.stats = {
            'total_predictions': 0,
            'uncertain_predictions': 0,
            'updates_performed': 0,
            'last_update_accuracy': None
        }
        
        # Checkpoint
        self.checkpoint_dir = Path('../../checkpoints/online')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def predict(self, x, return_confidence=False):
        """
        Tahmin yap ve düşük confidence örnekleri kaydet.
        """
        self.model.eval()
        x = x.to(self.device)
        
        with torch.no_grad():
            out = self.model(values=x, continuous=True, task='classification')
            probs = torch.softmax(out['base_output'], dim=-1)
            confidences = probs.max(dim=-1).values
            predictions = probs.argmax(dim=-1)
        
        # Düşük confidence olanları buffer'a ekle
        self.buffer.add_batch(x, confidences, predictions)
        
        # İstatistik güncelle
        self.stats['total_predictions'] += len(x)
        self.stats['uncertain_predictions'] += (confidences < 0.7).sum().item()
        
        if return_confidence:
            return predictions, confidences
        return predictions
    
    def check_and_update(self, validation_loader=None):
        """
        Güncelleme gerekli mi kontrol et, gerekliyse yap.
        """
        # Buffer'dan örnekleri al
        x, _ = self.buffer.get_samples()
        
        if x is None:
            return False, "Buffer empty"
        
        x = x.to(self.device)
        
        # Pseudo-label üret
        labels, mask, confidences = self.pseudo_labeler.get_pseudo_labels(self.model, x)
        reliable_count = mask.sum().item()
        
        # Validation accuracy hesapla
        val_accuracy = None
        if validation_loader is not None:
            val_accuracy = self._compute_validation_accuracy(validation_loader)
        
        # Güncelleme yapılmalı mı?
        should_update, reason = self.scheduler.should_update(
            self.buffer, reliable_count, len(x), val_accuracy
        )
        
        if not should_update:
            return False, reason
        
        # GÜNCELLEME YAP
        print(f"\n[OnlineLearner] Starting update...")
        print(f"  Buffer size: {len(self.buffer)}")
        print(f"  Reliable samples: {reliable_count}")
        
        # Checkpoint kaydet (rollback için)
        self._save_checkpoint('pre_update')
        
        # Güvenilir örnekleri al
        reliable_x, reliable_labels = self.pseudo_labeler.filter_reliable(x, labels, mask)
        
        if len(reliable_x) < 100:
            return False, f"Not enough reliable samples: {len(reliable_x)}"
        
        # Fine-tune yap
        success = self._fine_tune(reliable_x, reliable_labels, validation_loader)
        
        if success:
            self.scheduler.mark_updated()
            self.stats['updates_performed'] += 1
            self.buffer.clear()
            
            # Yeni EWC task kaydet
            dataset = TensorDataset(reliable_x, reliable_labels)
            dataloader = DataLoader(dataset, batch_size=64)
            self.ewc.register_new_task(dataloader, self.device)
            
            return True, "Update successful"
        else:
            # Rollback
            self._load_checkpoint('pre_update')
            return False, "Update failed, rolled back"
    
    def _fine_tune(self, x, labels, validation_loader=None, epochs=3, lr=0.0001):
        """
        Model'i fine-tune et (EWC ile).
        """
        self.model.train()
        
        # Önceki validation accuracy
        if validation_loader:
            prev_val_acc = self._compute_validation_accuracy(validation_loader)
        
        # Dataset hazırla
        dataset = TensorDataset(x, labels)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Optimizer (düşük LR)
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for bx, by in dataloader:
                bx, by = bx.to(self.device), by.to(self.device)
                
                optimizer.zero_grad()
                
                out = self.model(values=bx, continuous=True, task='classification')
                
                # Classification loss + EWC penalty
                ce_loss = loss_fn(out['base_output'], by)
                ewc_loss = self.ewc.penalty()
                loss = ce_loss + ewc_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                correct += (out['base_output'].argmax(-1) == by).sum().item()
                total += len(by)
            
            acc = 100 * correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}, Acc={acc:.1f}%")
        
        # Validation kontrolü
        if validation_loader:
            new_val_acc = self._compute_validation_accuracy(validation_loader)
            self.stats['last_update_accuracy'] = new_val_acc
            
            # Validation düştüyse başarısız
            if new_val_acc < prev_val_acc - 0.02:  # %2'den fazla düşüş
                print(f"  [WARNING] Validation dropped: {prev_val_acc:.2f} -> {new_val_acc:.2f}")
                return False
            
            print(f"  Validation: {prev_val_acc:.2f} -> {new_val_acc:.2f}")
        
        self.model.eval()
        return True
    
    def _compute_validation_accuracy(self, validation_loader):
        """Validation accuracy hesapla"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for bx, by in validation_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = self.model(values=bx, continuous=True, task='classification')
                correct += (out['base_output'].argmax(-1) == by).sum().item()
                total += len(by)
        
        return correct / total if total > 0 else 0
    
    def _save_checkpoint(self, name):
        """Checkpoint kaydet"""
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ewc_state': {
                'fisher': self.ewc.fisher,
                'old_params': self.ewc.old_params,
                'initialized': self.ewc.initialized,
                'task_count': self.ewc.task_count
            },
            'stats': self.stats
        }, path)
    
    def _load_checkpoint(self, name):
        """Checkpoint yükle"""
        path = self.checkpoint_dir / f'{name}.pt'
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])
        self.ewc.fisher = state['ewc_state']['fisher']
        self.ewc.old_params = state['ewc_state']['old_params']
        self.ewc.initialized = state['ewc_state']['initialized']
        self.ewc.task_count = state['ewc_state']['task_count']
        self.stats = state['stats']
    
    def get_stats(self):
        """İstatistikleri döndür"""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'ewc_tasks': self.ewc.task_count
        }
    
    def save(self, path):
        """Tüm sistemi kaydet"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'ewc_state': {
                'fisher': self.ewc.fisher,
                'old_params': self.ewc.old_params,
                'initialized': self.ewc.initialized,
                'task_count': self.ewc.task_count
            },
            'buffer': list(self.buffer.buffer),
            'stats': self.stats,
            'scheduler_last_update': self.scheduler.last_update
        }, path)
        print(f"[OnlineLearner] Saved to {path}")
    
    def load(self, path):
        """Tüm sistemi yükle"""
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])
        self.ewc.fisher = state['ewc_state']['fisher']
        self.ewc.old_params = state['ewc_state']['old_params']
        self.ewc.initialized = state['ewc_state']['initialized']
        self.ewc.task_count = state['ewc_state']['task_count']
        self.buffer.buffer = deque(state['buffer'], maxlen=self.buffer.buffer.maxlen)
        self.stats = state['stats']
        self.scheduler.last_update = state['scheduler_last_update']
        print(f"[OnlineLearner] Loaded from {path}")


# Kullanım örneği
if __name__ == "__main__":
    print("Online Learning System")
    print("=" * 50)
    print("Kullanım:")
    print("""
    from online_learning import OnlineLearner
    from model import TabularFoundationModel
    
    # Model yükle
    model = TabularFoundationModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Online learner başlat
    learner = OnlineLearner(model, config)
    
    # Tahmin yap (düşük confidence olanlar otomatik kaydedilir)
    predictions = learner.predict(new_data)
    
    # Periyodik olarak güncelleme kontrolü
    success, msg = learner.check_and_update(validation_loader)
    print(f"Update: {success}, {msg}")
    
    # İstatistikler
    print(learner.get_stats())
    """)
