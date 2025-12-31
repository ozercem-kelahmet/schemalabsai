import torch
import torch.nn as nn
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import time

from model import SchemalabsBaseModel12v1
from config import get_config

BATCH_SIZE = 512
EPOCHS = 10
LR = 0.001
SAMPLES_PER_SUBSECTOR = 5000
MISSING_RATE = 0.0

MODEL_NAME = "SchemalabsBaseModel12v1"

sectors = [
    "healthcare", "finance", "manufacturing", "logistics", "retail",
    "telecom", "insurance", "hr", "realestate", "ecommerce",
    "sports", "technology", "energy", "agriculture", "education",
    "entertainment", "hospitality", "automotive", "aerospace", "pharma",
    "biotech", "construction", "mining", "oil_gas", "utilities",
    "transport", "media", "advertising", "legal", "consulting",
    "security", "environmental", "food_beverage", "textiles", "chemicals",
    "metals", "electronics", "machinery", "defense", "maritime",
    "aviation", "railways", "postal", "warehousing", "packaging",
    "printing", "recycling", "water", "waste_mgmt", "renewable",
    "unknown"
]

feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                'recovery_index', 'response_rate']

print("="*70)
print(f"{MODEL_NAME} - FULL TRAINING")
print("+ MIDAS + MIRAS (12 Features) + Online Learning Ready")
print(f"5K samples/subsector | {EPOCHS} epochs | Batch {BATCH_SIZE}")
print("="*70)

print("\nBuilding maps...")
subsector_to_id = {}
label_to_id = {}
offset_map = {}
sub_idx = 0
lbl_idx = 0
global_offset = 0
total_batches = 0

for sector in sectors:
    path = Path(f'../../data/training_50x50/{sector}.parquet')
    if not path.exists():
        continue
    
    if sector == 'unknown':
        subsector_to_id['unknown_unknown'] = sub_idx
        offset_map['unknown_unknown'] = global_offset
        global_offset += 100
        sub_idx += 1
        label_to_id['unknown_unknown_unknown'] = lbl_idx
        lbl_idx += 1
        total_batches += (SAMPLES_PER_SUBSECTOR * 50) // BATCH_SIZE
    else:
        df = pd.read_parquet(path, columns=['subsector', 'target'])
        n_subs = len(df['subsector'].unique())
        total_batches += (n_subs * SAMPLES_PER_SUBSECTOR) // BATCH_SIZE
        for sub in sorted(df['subsector'].unique()):
            key = f"{sector}_{sub}"
            subsector_to_id[key] = sub_idx
            offset_map[key] = global_offset
            global_offset += 100
            sub_idx += 1
            for tgt in sorted(df[df['subsector']==sub]['target'].unique()):
                label_to_id[f"{sector}_{sub}_{tgt}"] = lbl_idx
                lbl_idx += 1
        del df
        gc.collect()

n_classes = len(label_to_id)
n_subsectors = len(subsector_to_id)
total_batches_all = total_batches * EPOCHS
print(f"Classes: {n_classes}, Subsectors: {n_subsectors}")
print(f"Total batches: {total_batches:,}/epoch, {total_batches_all:,} total")

config = get_config()
config['n_classes'] = n_classes
config['n_features'] = len(feature_cols)

model = SchemalabsBaseModel12v1(config)
print(f"Model: {MODEL_NAME}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Test model
print("\nTesting model...")
test_input = torch.randn(2, len(feature_cols))
try:
    test_out = model(values=test_input, continuous=True, task='classification')
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_out['base_output'].shape}")
    print(f"  Memory state shape: {test_out['memory_state'].shape}")
    print(f"  MIDAS loss: {test_out['midas_loss'].item():.4f}")
    print("  ✓ Model test PASSED")
except Exception as e:
    print(f"  ✗ Model test FAILED: {e}")
    sys.exit(1)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()

print("="*70)
sys.stdout.flush()

start_time = time.time()
global_step = 0
global_correct = 0
global_total = 0
persistent_memory = None

for epoch in range(EPOCHS):
    epoch_correct, epoch_total = 0, 0
    
    for sector in sectors:
        path = Path(f'../../data/training_50x50/{sector}.parquet')
        if not path.exists():
            continue
        
        df = pd.read_parquet(path)
        if sector != 'unknown':
            df = df.groupby('subsector', group_keys=False).head(SAMPLES_PER_SUBSECTOR)
        else:
            df = df.head(SAMPLES_PER_SUBSECTOR * 50)
        
        X = df[feature_cols].values.astype(np.float32)
        
        if sector == 'unknown':
            offset = offset_map['unknown_unknown']
            X = X + offset
            y = np.full(len(df), label_to_id['unknown_unknown_unknown'], dtype=np.int64)
        else:
            offsets = np.array([offset_map[f"{sector}_{s}"] for s in df['subsector']])
            X = X + offsets.reshape(-1, 1)
            y = np.array([label_to_id[f"{sector}_{s}_{t}"] for s, t in zip(df['subsector'], df['target'])], dtype=np.int64)
        
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        
        del df
        gc.collect()
        
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        
        n_batches = len(X) // BATCH_SIZE
        
        for i in range(n_batches):
            s = i * BATCH_SIZE
            bx = torch.FloatTensor(X[s:s+BATCH_SIZE])
            by = torch.LongTensor(y[s:s+BATCH_SIZE])
            
            if MISSING_RATE > 0:
                missing_mask = (torch.rand_like(bx) < MISSING_RATE).float()
            else:
                missing_mask = None
            
            optimizer.zero_grad()
            out = model(values=bx, continuous=True, task='classification', 
                       prev_memory=persistent_memory, missing_mask=missing_mask)
            
            persistent_memory = out['memory_state'].detach()
            
            loss = loss_fn(out['base_output'], by)
            if MISSING_RATE > 0:
                loss = loss + 0.1 * out['midas_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            correct = (out['base_output'].argmax(-1) == by).sum().item()
            epoch_correct += correct
            epoch_total += BATCH_SIZE
            global_correct += correct
            global_total += BATCH_SIZE
            global_step += 1
        
        pct = 100 * global_step / total_batches_all
        elapsed = (time.time() - start_time) / 3600
        eta = (elapsed / global_step) * (total_batches_all - global_step) if global_step > 0 else 0
        acc = 100 * global_correct / global_total
        
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        sys.stdout.write(f"\r\033[K[{bar}] {pct:.1f}% | E{epoch+1}/{EPOCHS} | {sector:<12} | Acc:{acc:.1f}% | ETA:{eta:.1f}h")
        sys.stdout.flush()
        
        del X, y
        gc.collect()
    
    scheduler.step()
    epoch_acc = 100 * epoch_correct / epoch_total
    print(f"\n>>> Epoch {epoch+1}: {epoch_acc:.1f}%")
    
    torch.save({
        'model_name': MODEL_NAME,
        'model_state_dict': model.state_dict(),
        'config': config,
        'subsector_to_id': subsector_to_id,
        'label_to_id': label_to_id,
        'offset_map': offset_map,
        'n_classes': n_classes,
        'n_subsectors': n_subsectors,
        'epoch': epoch + 1,
        'accuracy': epoch_acc
    }, f'../../checkpoints/{MODEL_NAME}_epoch{epoch+1}.pt')

total_time = (time.time() - start_time) / 3600
print(f"\n{'='*70}")
print(f"DONE: {epoch_acc:.1f}% | {total_time:.1f}h")
print(f"SAVED: {MODEL_NAME}_full.pt")

torch.save({
    'model_name': MODEL_NAME,
    'model_state_dict': model.state_dict(),
    'config': config,
    'subsector_to_id': subsector_to_id,
    'label_to_id': label_to_id,
    'offset_map': offset_map,
    'n_classes': n_classes,
    'n_subsectors': n_subsectors,
    'epochs': EPOCHS,
    'accuracy': epoch_acc,
    'features': {
        'midas': True,
        'miras': True,
        'persistent_memory': True,
        'gradient_clipping': True,
        'online_learning_ready': True
    }
}, f'../../checkpoints/{MODEL_NAME}_full.pt')
