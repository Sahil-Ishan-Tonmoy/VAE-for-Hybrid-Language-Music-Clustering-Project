"""
1D Convolutional Variational Autoencoder for MFCC Features
Treats 20-dimensional MFCC vectors as 1D sequences for convolutional processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    """
    1D Convolutional VAE for MFCC feature vectors
    
    Architecture:
    - Encoder: Conv1D layers to extract hierarchical features
    - Latent: 16-dimensional latent space
    - Decoder: Transposed Conv1D layers for reconstruction
    """
    
    def __init__(self, input_dim=20, latent_dim=16, beta=1.0):
        super(ConvVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: 1D Convolutions over MFCC features
        # Input: (batch, 1, 20) - treating 20 MFCCs as sequence
        self.encoder = nn.Sequential(
            # Layer 1: 20 -> 20
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2: 20 -> 10
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: 10 -> 5
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # Flattened size: 64 channels × 5 positions = 320
        self.flatten_size = 64 * 5
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input projection
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder: Transposed Conv1D for upsampling
        self.decoder = nn.Sequential(
            # Layer 1: 5 -> 10
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2: 10 -> 20
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: 20 -> 20 (reconstruction)
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        # Reshape for Conv1D: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Convolutional encoding
        h = self.encoder(x)
        
        # Flatten
        h = h.view(h.size(0), -1)
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        
        Returns:
            z: (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction
        
        Args:
            z: (batch_size, latent_dim)
        
        Returns:
            x_recon: (batch_size, input_dim)
        """
        # Project to decoder input size
        h = self.fc_decode(z)
        
        # Reshape for Conv1D: (batch, flatten_size) -> (batch, 64, 5)
        h = h.view(h.size(0), 64, 5)
        
        # Transposed convolutions for upsampling
        x_recon = self.decoder(h)
        
        # Reshape: (batch, 1, 20) -> (batch, 20)
        x_recon = x_recon.squeeze(1)
        
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through Conv VAE
        
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            x_recon: (batch_size, input_dim)
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        """
        VAE loss = Reconstruction loss + Beta * KL divergence
        
        Args:
            x: (batch_size, input_dim) - original input
            x_recon: (batch_size, input_dim) - reconstruction
            mu: (batch_size, latent_dim) - latent mean
            logvar: (batch_size, latent_dim) - latent log variance
        
        Returns:
            loss: scalar
            recon_loss: scalar
            kl_loss: scalar
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with beta weighting
        loss = recon_loss + self.beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def get_latent_features(self, x):
        """
        Extract latent features (used for clustering)
        
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            z: (batch_size, latent_dim)
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            # Use mean (deterministic) for clustering
            return mu


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("1D Convolutional VAE Architecture Test")
    print("="*70)
    
    # Create model
    model = ConvVAE(input_dim=20, latent_dim=16, beta=1.0)
    
    # Model summary
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"\nArchitecture:")
    print(model)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 20)
    
    print(f"\n{'='*70}")
    print("Forward Pass Test")
    print(f"{'='*70}")
    print(f"Input shape: {x.shape}")
    
    x_recon, mu, logvar = model(x)
    
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss computation
    loss, recon_loss, kl_loss = model.loss_function(x, x_recon, mu, logvar)
    print(f"\nLoss computation:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence: {kl_loss.item():.4f}")
    
    # Test latent extraction
    latent = model.get_latent_features(x)
    print(f"\nLatent features shape: {latent.shape}")
    
    print(f"\n{'='*70}")
    print("✅ Conv VAE architecture test passed!")
    print(f"{'='*70}")
