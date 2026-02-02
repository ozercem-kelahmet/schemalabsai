# model_base.py'deki update_heads'i düzelt

import re

with open('/Users/ozercemkelahmet/Desktop/schemalabsai/model/model_base.py', 'r') as f:
    content = f.read()

old = '''    def update_heads(self, n_classes=None, n_features=None, n_sectors=None):
        if n_classes:
            self.n_classes = n_classes
            self.final_head = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, n_classes)
            )
        if n_sectors:
            self.n_sectors = n_sectors
            self.sector_head = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, n_sectors)
            )
        if n_features:
            self.n_features = n_features
            self.values_proj = nn.Sequential(
                nn.Linear(n_features, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU()
            )'''

new = '''    def update_heads(self, n_classes=None, n_features=None, n_sectors=None):
        if n_classes and n_classes != self.n_classes:
            self.n_classes = n_classes
            self.final_head = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, n_classes)
            )
        if n_sectors and n_sectors != self.n_sectors:
            self.n_sectors = n_sectors
            self.sector_head = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, n_sectors)
            )
        if n_features and n_features != self.n_features:
            self.n_features = n_features
            self.values_proj = nn.Sequential(
                nn.Linear(n_features, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU()
            )'''

content = content.replace(old, new)

with open('/Users/ozercemkelahmet/Desktop/schemalabsai/model/model_base.py', 'w') as f:
    f.write(content)

print("Fixed!")
