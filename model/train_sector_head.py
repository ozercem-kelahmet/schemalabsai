"""
SchemaLabs - SectorHead Training
column names + value stats → sector tahmin
"""

import json, re, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE  = Path("data/sector_labels.json")
SAVE_PATH  = Path("data/sector_head.pt")
D_MODEL    = 128
MAX_COLS   = 50
MAX_TOKENS = 5
EPOCHS     = 300
BATCH_SIZE = 8
LR         = 0.0003

def tokenize(name):
    name = name.lower()
    name = re.sub(r'[_\-\s\.]', ' ', name)
    return name.split()

def build_vocab(items):
    vocab = {"<pad>": 0, "<unk>": 1}
    for item in items:
        for col in item["column_names"]:
            for tok in tokenize(col):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab

def encode_columns(column_names, vocab):
    matrix = np.zeros((MAX_COLS, MAX_TOKENS), dtype=np.int64)
    for i, col in enumerate(column_names[:MAX_COLS]):
        for j, tok in enumerate(tokenize(col)[:MAX_TOKENS]):
            matrix[i][j] = vocab.get(tok, 1)
    return matrix

def encode_stats(column_stats):
    matrix = np.zeros((MAX_COLS, 6), dtype=np.float32)
    for i, (col, s) in enumerate(list(column_stats.items())[:MAX_COLS]):
        matrix[i][0] = float(s.get("null_ratio", 0))
        matrix[i][1] = float(s.get("unique_ratio", 0))
        matrix[i][2] = 1.0 if "mean" in s else 0.0
        matrix[i][3] = float(np.tanh(s.get("mean", 0) / 100.0)) if "mean" in s else 0.0
        matrix[i][4] = float(np.tanh(s.get("std",  0) / 100.0)) if "std"  in s else 0.0
        matrix[i][5] = float(np.tanh(s.get("max",  0) / 100.0)) if "max"  in s else 0.0
    return matrix

class SectorDataset(Dataset):
    def __init__(self, items, vocab, le):
        self.items = items
        self.vocab = vocab
        self.le    = le

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item       = self.items[idx]
        col_tokens = encode_columns(item["column_names"], self.vocab)
        col_stats  = encode_stats(item["features"]["column_stats"])
        label      = self.le.transform([item["sector"]])[0]
        return (
            torch.tensor(col_tokens, dtype=torch.long),
            torch.tensor(col_stats,  dtype=torch.float32),
            torch.tensor(label,      dtype=torch.long)
        )

class SectorHead(nn.Module):
    def __init__(self, vocab_size, n_sectors, d_model=D_MODEL):
        super().__init__()
        self.token_embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.col_encoder  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.1))
        self.stats_encoder= nn.Sequential(nn.Linear(6, d_model), nn.ReLU(), nn.Dropout(0.1))
        self.col_attn     = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.classifier   = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_model, n_sectors)
        )

    def forward(self, col_tokens, col_stats):
        emb      = self.token_embed(col_tokens).mean(dim=2)   # B, C, d
        col_emb  = self.col_encoder(emb)
        stats_emb= self.stats_encoder(col_stats)
        combined = col_emb + stats_emb
        attn_out, _ = self.col_attn(combined, combined, combined)
        mean_pool= attn_out.mean(dim=1)
        max_pool = attn_out.max(dim=1).values
        return self.classifier(torch.cat([mean_pool, max_pool], dim=-1))

def train():
    data  = json.loads(DATA_FILE.read_text())
    items = list(data.values())
    print(f"Dataset: {len(items)}")

    le = LabelEncoder()
    le.fit([it["sector"] for it in items])
    n_sectors = len(le.classes_)
    print(f"Sektör: {n_sectors} → {list(le.classes_)}")

    vocab = build_vocab(items)
    print(f"Vocab: {len(vocab)}")

    # debug: ilk item'ın column_names'ini göster
    print(f"Örnek item column_names: {items[0]['column_names'][:5]}")
    print(f"Örnek encoded tokens[0]: {encode_columns(items[0]['column_names'], vocab)[0]}")

    train_items, val_items = train_test_split(items, test_size=0.15, random_state=42)
    print(f"Train: {len(train_items)}, Val: {len(val_items)}")

    train_ds = SectorDataset(train_items, vocab, le)
    val_ds   = SectorDataset(val_items,   vocab, le)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = SectorHead(vocab_size=len(vocab), n_sectors=n_sectors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = nn.CrossEntropyLoss()

    best_val_acc = 0
    print("\nTraining...\n" + "="*50)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for col_tokens, col_stats, labels in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(col_tokens, col_stats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for col_tokens, col_stats, labels in val_dl:
                pred = model(col_tokens, col_stats).argmax(-1)
                correct += (pred == labels).sum().item()
                total   += len(labels)

        val_acc = correct / total * 100
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "label_encoder_classes": le.classes_.tolist(),
                "d_model": D_MODEL,
                "max_cols": MAX_COLS,
            }, SAVE_PATH)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | loss={train_loss/len(train_dl):.4f} | val={val_acc:.1f}% | best={best_val_acc:.1f}%")

    print(f"\nBest val acc: {best_val_acc:.1f}% → {SAVE_PATH}")

if __name__ == "__main__":
    train()
