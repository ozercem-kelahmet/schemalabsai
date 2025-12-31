import torch
import torch.nn as nn
import torch.nn.functional as F

class MIDAS(nn.Module):
    """Missing Data Imputation via Denoising Autoencoder"""
    
    def __init__(self, d_input, d_hidden=128):
        super().__init__()
        self.d_input = d_input
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden // 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_hidden // 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_input)
        )
        
    def forward(self, x, mask=None):
        """
        x: input tensor [batch, features]
        mask: 1 = missing, 0 = present [batch, features]
        """
        if mask is None:
            return x, torch.zeros(1, device=x.device)
        
        # Eksik değerleri 0 yap
        x_masked = x * (1 - mask)
        
        # Encode -> Decode
        z = self.encoder(x_masked)
        x_reconstructed = self.decoder(z)
        
        # Sadece eksik yerleri doldur
        x_imputed = x * (1 - mask) + x_reconstructed * mask
        
        # Reconstruction loss (sadece eksik yerler için)
        if self.training:
            loss = F.mse_loss(x_reconstructed * mask, x * mask, reduction='sum')
            loss = loss / (mask.sum() + 1e-8)
        else:
            loss = torch.zeros(1, device=x.device)
        
        return x_imputed, loss
