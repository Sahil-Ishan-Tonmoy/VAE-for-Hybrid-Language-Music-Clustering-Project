import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MusicVAE(nn.Module):
    """
    Variational Autoencoder for music feature clustering
    """
    
    def __init__(self, input_dim=20, hidden_dim=128, latent_dim=16):
        """
        Args:
            input_dim: Dimension of input features (MFCC features)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
        """
        super(MusicVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Latent space layers (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def encode(self, x):
        """
        Encode input to latent space parameters
        
        Args:
            x: Input features
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction
        
        Args:
            z: Latent vector
            
        Returns:
            reconstruction: Reconstructed input
        """
        h = F.relu(self.bn3(self.fc3(z)))
        h = self.dropout(h)
        h = F.relu(self.bn4(self.fc4(h)))
        reconstruction = self.fc5(h)
        
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x: Input features
            
        Returns:
            reconstruction: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def get_latent(self, x):
        """
        Get latent representation (just the mean, no sampling)
        
        Args:
            x: Input features
            
        Returns:
            mu: Mean of latent distribution
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence
    
    Args:
        reconstruction: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (beta > 1 for Beta-VAE)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
    
    # KL divergence loss
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class MusicDataset(Dataset):
    """
    PyTorch Dataset for music features
    """
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_vae(model, train_loader, optimizer, device, beta=1.0):
    """
    Train VAE for one epoch
    
    Args:
        model: VAE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device (cpu/cuda)
        beta: Beta parameter for loss
        
    Returns:
        avg_loss: Average loss for epoch
    """
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction, mu, logvar = model(data)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader.dataset)
    return avg_loss


def extract_latent_features(model, data_loader, device):
    """
    Extract latent features from trained VAE
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device (cpu/cuda)
        
    Returns:
        latent_features: Latent representations
        labels: Corresponding labels
    """
    model.eval()
    latent_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_features.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return latent_features, all_labels


if __name__ == "__main__":
    # Example usage
    print("VAE Model Implementation")
    print("=" * 60)
    
    # Create dummy data for testing
    input_dim = 20
    batch_size = 32
    n_samples = 100
    
    dummy_features = np.random.randn(n_samples, input_dim)
    dummy_labels = np.random.randint(0, 10, n_samples)
    
    # Create dataset and dataloader
    dataset = MusicDataset(dummy_features, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicVAE(input_dim=input_dim, hidden_dim=128, latent_dim=16).to(device)
    
    print(f"Model created on device: {device}")
    print(f"Model architecture:")
    print(model)
    print("\n" + "=" * 60)
    
    # Test forward pass
    sample_data, _ = next(iter(dataloader))
    sample_data = sample_data.to(device)
    
    reconstruction, mu, logvar = model(sample_data)
    print(f"Input shape: {sample_data.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    print("\nVAE model test successful!")