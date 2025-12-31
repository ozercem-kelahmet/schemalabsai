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
EPOCHS = 1
LR = 0.001
SAMPLES_PER_SUBSECTOR = 5000

MODEL_NAME = "SchemalabsBaseModel12v1_fix"

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
    "printing", "recycling", "water", "waste_mgmt", "renewable"
]

feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                'recovery_index', 'response_rate']

print("="*70)
print(f"{MODEL_NAME} - 1 EPOCH TEST")
print("="*70)

print("\nBuilding maps...")
subsector_to_id = {}
label_to_id = {}
sub_idx = 0
lbl_idx = 0
total_batches = 0

for sector in sectors:
    path = Path(f'../../data/training_50x50/{sector}.parquet')
    if not path.exists():
        continue
    
    df = pd.read_parquet(path, columns=['subsector', 'target'])
    n_subs = len(df['subsector'].unique())
    total_batches += (n_subs * SAMPLES_PER_SUBSECTOR) // BATCH_SIZE
    
    for sub in sorted(df['subsector'].unique()):
        key = f"{sector}_{sub}"
        subsector_to_id[key] = sub_idx
        sub_idx += 1
        for tgt in sorted(df[df['subsector']==sub]['target'].unique()):
            label_to_id[f"{sector}_{sub}_{tgt}"] = lbl_idx
            lbl_idx += 1
    del df
    gc.collect()

n_classes = len(label_to_id)
print(f"Classes: {n_classes}, Subsectors: {len(subsector_to_id)}")

config = get_config()
config['n_classes'] = n_classes
config['n_features'] = len(feature_cols)

model = SchemalabsBaseModel12v1(config)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

print("="*70)

start_time = time.time()
global_correct = 0
global_total = 0

for sector in sectors:
    path = Path(f'../../data/training_50x50/{sector}.parquet')
    if not path.exists():
        continue
    
    df = pd.read_parquet(path)
    df = df.groupby('subsector', group_keys=False).head(SAMPLES_PER_SUBSECTOR)
    
    X = df[feature_cols].values.astype(np.float32)
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
        
        optimizer.zero_grad()
        out = model(values=bx, continuous=True, task='classification')
        
        loss = loss_fn(out['base_output'], by)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        correct = (out['base_output'].argmax(-1) == by).sum().item()
        global_correct += correct
        global_total += BATCH_SIZE
    
    acc = 100 * global_correct / global_total
    sys.stdout.write(f"\r{sector:<15} | Acc: {acc:.1f}%")
    sys.stdout.flush()
    
    del X, y
    gc.collect()

print(f"\n\n>>> Epoch 1: {acc:.1f}%")

# Football test
print("\nFootball test:")
model.eval()
test_df = pd.read_parquet('../../data/training_50x50/sports.parquet')
test_df = test_df[test_df['subsector'] == 'football'].head(100)
X_test = test_df[feature_cols].values.astype(np.float32)
X_test = (X_test - X_test.mean(0)) / (X_test.std(0) + 1e-8)
y_test = [label_to_id[f"sports_{s}_{t}"] for s, t in zip(test_df['subsector'], test_df['target'])]

with torch.no_grad():
    out = model(values=torch.FloatTensor(X_test), continuous=True, task='classification')
    preds = out['base_output'].argmax(-1).tolist()

test_acc = sum(1 for p, t in zip(preds, y_test) if p == t)
print(f"Football: {test_acc}/100 = {test_acc}%")

torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'label_to_id': label_to_id,
    'n_classes': n_classes,
}, f'../../checkpoints/{MODEL_NAME}.pt')

print(f"\nDONE: {(time.time()-start_time)/3600:.1f}h")
