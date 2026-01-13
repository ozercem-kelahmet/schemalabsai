import torch
import torch.nn as nn

class SchemaProcessing(nn.Module):
    """Kolon isimlerinden anlam çıkarır ve embedding üretir"""
    def __init__(self, d_model, n_heads=8, n_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # Karakter bazlı kolon ismi encoding
        self.char_embed = nn.Embedding(256, d_model)  # ASCII karakterler
        self.col_encoder = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads,
                dim_feedforward=d_model * 2,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Projection for pre-computed embeddings
        self.proj = nn.Linear(d_model, d_model)
    
    def encode_column_names(self, col_names):
        """Kolon isimlerini embedding'e çevir"""
        embeddings = []
        device = self.char_embed.weight.device
        
        for col_name in col_names:
            # Karakter bazlı encoding (max 32 karakter)
            char_ids = torch.tensor([ord(c) % 256 for c in str(col_name)[:32]], device=device)
            char_emb = self.char_embed(char_ids).unsqueeze(0)  # (1, seq_len, d_model)
            
            # LSTM ile özetle
            _, (h_n, _) = self.col_encoder(char_emb)
            col_emb = torch.cat([h_n[0], h_n[1]], dim=-1)  # (1, d_model)
            embeddings.append(col_emb)
        
        return torch.cat(embeddings, dim=0)  # (n_cols, d_model)
    
    def forward(self, schema_info, batch_size=None):
        """
        Args:
            schema_info: kolon isimleri (list of str) veya pre-computed embedding (tensor)
            batch_size: batch boyutu (kolon isimleri verildiğinde gerekli)
        """
        # Eğer string listesi ise, encode et
        if isinstance(schema_info, (list, tuple)) and len(schema_info) > 0 and isinstance(schema_info[0], str):
            x = self.encode_column_names(schema_info)
            if batch_size is not None:
                x = x.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                x = x.unsqueeze(0)
        else:
            # Pre-computed embedding
            x = self.proj(schema_info)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)
