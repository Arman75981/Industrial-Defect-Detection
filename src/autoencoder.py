"""
Lightweight Convolutional Autoencoder for Visual Defect Detection.
Designed for CPU execution with small memory footprint and fast inference.
"""

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Lightweight Convolutional Autoencoder.
    
    Architecture:
      Encoder:
        - Input: (B, 3, 128, 128)
        - Layer 1: Conv2D(3->32) + BatchNorm + LeakyReLU + MaxPool -> (B, 32, 64, 64)
        - Layer 2: Conv2D(32->64) + BatchNorm + LeakyReLU + MaxPool -> (B, 64, 32, 32)
        - Layer 3: Conv2D(64->128) + BatchNorm + LeakyReLU + MaxPool -> (B, 128, 16, 16)
        - Layer 4: Conv2D(128->256) + BatchNorm + LeakyReLU + MaxPool -> (B, 256, 8, 8)
        
      Bottleneck (Latent Space Representation):
        - Layer: Conv2D(256->64) + BatchNorm + LeakyReLU -> (B, 64, 8, 8)
        
      Decoder:
        - Layer 1: ConvTranspose2d(64->128, stride=2) + BatchNorm + LeakyReLU -> (B, 128, 16, 16)
        - Layer 2: ConvTranspose2d(128->64, stride=2) + BatchNorm + LeakyReLU -> (B, 64, 32, 32)
        - Layer 3: ConvTranspose2d(64->32, stride=2) + BatchNorm + LeakyReLU -> (B, 32, 64, 64)
        - Layer 4: ConvTranspose2d(32->16, stride=2) + BatchNorm + LeakyReLU -> (B, 16, 128, 128)
        - Output: Conv2D(16->3, kernel=3) + Sigmoid -> (B, 3, 128, 128)
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Encoder: Extracts hierarchical features and compresses spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        # Bottleneck: Forces the network to learn essential normal structural patterns
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder: Reconstructs the original normal image from the compact latent features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # 64x64 -> 128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Bound outputs strictly to [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        bottleneck = self.bottleneck(latent)
        reconstruction = self.decoder(bottleneck)
        return reconstruction

def get_autoencoder(device: torch.device = torch.device("cpu")) -> ConvAutoencoder:
    """Instantiates and returns the ConvAutoencoder on the specified device."""
    model = ConvAutoencoder().to(device)
    return model
