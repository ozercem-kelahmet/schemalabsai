import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import sys
sys.path.append('..')

from model import TabularFoundationModel
from config import get_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_sector(name, target_col, n_samples=50000):
    df = pd.read_parquet(f'../../data/training_subset/{name}_1M.parquet').head(n_samples)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    features = df[num_cols].values.astype(np.float32)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    le = LabelEncoder()
    targets = le.fit_transform(df[target_col])
    
    return features, targets, len(le.classes_), scaler, le, num_cols

def main():
    print("="*60)
    print("12 SECTOR TRAINING - 600K")
    print("="*60)
    
    device = 'cpu'
    
    sectors = [
        ('healthcare', 'disease'),
        ('finance', 'is_fraud'),
        ('manufacturing', 'quality'),
        ('ecommerce', 'category'),
        ('telecom', 'churn'),
        ('logistics', 'status'),
        ('insurance', 'risk_level'),
        ('retail', 'segment'),
        ('hr', 'left'),
        ('realestate', 'property_type'),
        ('technology', 'product_type'),
        ('sports', 'sport_type')
    ]
    
    all_features = []
    all_targets = []
    all_scalers = {}
    all_encoders = {}
    total_classes = 0
    class_offsets = {}
    
    for name, target_col in sectors:
        features, targets, n_classes, scaler, le, num_cols = load_sector(name, target_col)
        
        n_cols = features.shape[1]
        if n_cols < 10:
            pad = np.zeros((features.shape[0], 10 - n_cols))
            features = np.hstack([features, pad])
        elif n_cols > 10:
            features = features[:, :10]
        
        class_offsets[name] = total_classes
        targets_adjusted = targets + total_classes
        total_classes += n_classes
        
        all_features.append(features)
        all_targets.append(targets_adjusted)
        all_scalers[name] = scaler
        all_encoders[name] = le
        
        print(f"{name}: {len(features)} samples, {n_classes} classes")
    
    features = np.vstack(all_features)
    targets = np.hstack(all_targets)
    
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    features = features[indices]
    targets = targets[indices]
    
    print(f"\nTotal: {len(features)} samples, {total_classes} classes")
    print("="*60)
    
    config = get_config()
    config['n_classes'] = total_classes
    config['max_cols'] = 10
    
    model = TabularFoundationModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    batch_size = 256
    num_samples = len(features)
    
    for epoch in range(5):
        idx_shuffle = np.arange(num_samples)
        np.random.shuffle(idx_shuffle)
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(range(0, num_samples - batch_size, batch_size), desc=f"Epoch {epoch+1}")
        
        for batch_num, idx in enumerate(pbar):
            batch_idx = idx_shuffle[idx:idx+batch_size]
            batch_features = torch.FloatTensor(features[batch_idx]).to(device)
            batch_targets = torch.LongTensor(targets[batch_idx]).to(device)
            
            optimizer.zero_grad()
            outputs = model(values=batch_features, continuous=True, task='classification')
            
            logits = outputs['base_output']
            loss = loss_fn(logits, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_targets).sum().item()
            total += batch_size
            
            avg_loss = total_loss / (batch_num + 1)
            acc = 100 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.1f}%'})
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'class_offsets': class_offsets,
        'scalers': all_scalers,
        'encoders': all_encoders,
        'sectors': [s[0] for s in sectors]
    }, '../../checkpoints/model_12sector.pt')
    
    print("\nSAVED: model_12sector.pt")
    
    print("\n" + "="*60)
    print("VALIDATION PER SECTOR")
    print("="*60)
    
    model.eval()
    
    for name, target_col in sectors:
        df_val = pd.read_parquet(f'../../data/training_subset/{name}_1M.parquet').tail(10000)
        num_cols = df_val.select_dtypes(include=['number']).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        
        val_features = all_scalers[name].transform(df_val[num_cols].values.astype(np.float32))
        
        n_cols = val_features.shape[1]
        if n_cols < 10:
            pad = np.zeros((val_features.shape[0], 10 - n_cols))
            val_features = np.hstack([val_features, pad])
        elif n_cols > 10:
            val_features = val_features[:, :10]
        
        val_targets = all_encoders[name].transform(df_val[target_col]) + class_offsets[name]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_features) - batch_size, batch_size):
                batch_f = torch.FloatTensor(val_features[i:i+batch_size])
                batch_t = val_targets[i:i+batch_size]
                outputs = model(values=batch_f, continuous=True, task='classification')
                preds = outputs['base_output'].argmax(dim=-1).numpy()
                correct += (preds == batch_t).sum()
                total += batch_size
        
        val_acc = 100 * correct / total
        print(f"{name}: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
