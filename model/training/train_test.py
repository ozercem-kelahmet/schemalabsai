import torch
import torch.nn as nn
from torch.optim import AdamW
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from pathlib import Path
import gc

from model import SchemalabsBaseModel12v1
from config import get_config

BATCH_SIZE = 512
LR = 0.001
SAMPLES = 1000  # Hızlı test için az sample

sectors = ["healthcare", "finance", "sports", "technology", "retail"]
feature_cols = ['primary_score', 'secondary_score', 'tertiary_score', 'risk_index', 
                'severity_level', 'duration_factor', 'frequency_rate', 'intensity_score',
                'recovery_index', 'response_rate']

print("Building maps...")
label_to_id = {}
lbl_idx = 0

for sector in sectors:
    df = pd.read_parquet(f'../../data/training_50x50/{sector}.parquet', columns=['subsector', 'target'])
    for sub in sorted(df['subsector'].unique())[:5]:  # İlk 5 subsector
        for tgt in sorted(df[df['subsector']==sub]['target'].unique()):
            label_to_id[f"{sector}_{sub}_{tgt}"] = lbl_idx
            lbl_idx += 1
    del df

n_classes = len(label_to_id)
print(f"Classes: {n_classes}")

config = get_config()
config['n_classes'] = n_classes
config['n_features'] = 10

model = SchemalabsBaseModel12v1(config)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

print("Training...")
total_correct = 0
total_samples = 0

for sector in sectors:
    df = pd.read_parquet(f'../../data/training_50x50/{sector}.parquet')
    df = df[df['subsector'].isin(sorted(df['subsector'].unique())[:5])]
    df = df.head(SAMPLES)
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Label oluştur
    y = []
    for s, t in zip(df['subsector'], df['target']):
        key = f"{sector}_{s}_{t}"
        if key in label_to_id:
            y.append(label_to_id[key])
    y = np.array(y[:len(X)], dtype=np.int64)
    X = X[:len(y)]
    
    # SADECE NORMALIZE - OFFSET YOK!
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    
    bx = torch.FloatTensor(X)
    by = torch.LongTensor(y)
    
    optimizer.zero_grad()
    out = model(values=bx, continuous=True, task='classification')
    loss = loss_fn(out['base_output'], by)
    loss.backward()
    optimizer.step()
    
    correct = (out['base_output'].argmax(-1) == by).sum().item()
    total_correct += correct
    total_samples += len(y)
    
    print(f"{sector}: {correct}/{len(y)} = {100*correct/len(y):.1f}%")

print(f"\nTotal: {total_correct}/{total_samples} = {100*total_correct/total_samples:.1f}%")

# Test
print("\nTest (sports/football):")
model.eval()
test_df = pd.read_parquet('../../data/training_50x50/sports.parquet')
test_df = test_df[test_df['subsector'] == 'football'].head(50)
X_test = test_df[feature_cols].values.astype(np.float32)
X_test = (X_test - X_test.mean(0)) / (X_test.std(0) + 1e-8)

y_test = []
for s, t in zip(test_df['subsector'], test_df['target']):
    key = f"sports_{s}_{t}"
    if key in label_to_id:
        y_test.append(label_to_id[key])

with torch.no_grad():
    out = model(values=torch.FloatTensor(X_test[:len(y_test)]), continuous=True, task='classification')
    preds = out['base_output'].argmax(-1).tolist()

correct = sum(1 for p, t in zip(preds, y_test) if p == t)
print(f"Football: {correct}/{len(y_test)} = {100*correct/len(y_test):.1f}%")
