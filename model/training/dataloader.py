import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder

class TabularDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.parquet_files = list(self.data_path.glob('*.parquet'))
        print(f"Found {len(self.parquet_files)} parquet files")
        
    def __len__(self):
        return len(self.parquet_files)
    
    def __getitem__(self, idx):
        df = pd.read_parquet(self.parquet_files[idx])
        
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        df = df.fillna(0)
        
        values = df.values.astype(np.float32)
        
        if values.shape[0] > 1000:
            values = values[:1000, :]
        elif values.shape[0] < 1000:
            pad = np.zeros((1000 - values.shape[0], values.shape[1]))
            values = np.vstack([values, pad])
        
        if values.shape[1] > 100:
            values = values[:, :100]
        elif values.shape[1] < 100:
            pad = np.zeros((values.shape[0], 100 - values.shape[1]))
            values = np.hstack([values, pad])
        
        sector = self.parquet_files[idx].stem.split('_')[0]
        
        return {
            'values': torch.FloatTensor(values),
            'sector': sector
        }

def create_dataloader(data_path, batch_size=4):
    dataset = TabularDataset(data_path)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # MEMORY LEAK FIX!
        pin_memory=False
    )
