import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from model import TabularFoundationModel
from config import get_config

def finetune_model(filepath, epochs=2):
    print(f"Loading data from {filepath}...")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type")
    
    df = df.select_dtypes(include=['number']).fillna(0)
    
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 columns (features + target)")
    
    print(f"Data shape: {df.shape}")
    
    config = get_config()
    model = TabularFoundationModel(config)
    
    checkpoint = torch.load('../checkpoints/model_37m_final.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Base model loaded, starting fine-tuning...")
    
    for name, param in model.named_parameters():
        if 'domain_adapters' not in name and 'task_heads' not in name:
            param.requires_grad = False
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        
        for i in range(0, len(df), 100):
            if i + 100 > len(df):
                break
            
            batch_df = df.iloc[i:i+100]
            
            data = batch_df.iloc[:, :-1].values.astype(np.float32)
            target = batch_df.iloc[:, -1].values.astype(np.float32)
            
            if data.shape[1] < 10:
                pad = np.zeros((data.shape[0], 10 - data.shape[1]))
                data = np.hstack([data, pad])
            else:
                data = data[:, :10]
            
            values = torch.FloatTensor(data).unsqueeze(0)
            values_flat = values.reshape(1, -1).long() % 50000
            types = torch.zeros_like(values_flat)
            
            col_sums = values.sum(dim=1)
            schema_info = col_sums.unsqueeze(2).repeat(1, 1, 512)
            
            domain_id = torch.zeros(1, dtype=torch.long)
            
            target_class = int(np.abs(target.mean()) % 10)
            target_tensor = torch.LongTensor([target_class])
            
            optimizer.zero_grad()
            
            outputs = model(
                values=values_flat,
                types=types,
                schema_info=schema_info,
                domain_id=domain_id,
                domain_name='default',
                task='classification'
            )
            
            loss = loss_fn(outputs['base_output'], target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'finetuned': True
    }, '../checkpoints/model_finetuned.pt')
    
    print("Fine-tuning complete!")
    return avg_loss
