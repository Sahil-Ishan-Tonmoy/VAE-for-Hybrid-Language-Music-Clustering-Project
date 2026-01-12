import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalVAE(nn.Module):
    """
    Multi-Modal VAE that fuses audio and lyrics features
    """

    def __init__(self, audio_dim=20, lyrics_dim=30, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.audio_dim = audio_dim
        self.lyrics_dim = lyrics_dim
        self.input_dim = audio_dim + lyrics_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, hidden_dim)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.fc_audio = nn.Linear(hidden_dim, audio_dim)
        self.fc_lyrics = nn.Linear(hidden_dim, lyrics_dim)
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = torch.relu(self.bn2(self.fc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.bn3(self.fc3(z)))
        h = self.dropout(h)
        h = torch.relu(self.bn4(self.fc4(h)))
        return self.fc_audio(h), self.fc_lyrics(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        audio_recon, lyrics_recon = self.decode(z)
        recon = torch.cat([audio_recon, lyrics_recon], dim=1)
        return recon, mu, logvar

    def loss_function(self, x, recon_x, mu, logvar, beta=1.0):
        audio_x = x[:, :self.audio_dim]
        lyrics_x = x[:, self.audio_dim:]
        audio_recon = recon_x[:, :self.audio_dim]
        lyrics_recon = recon_x[:, self.audio_dim:]

        recon_loss = (
            F.mse_loss(audio_recon, audio_x, reduction="sum")
            + F.mse_loss(lyrics_recon, lyrics_x, reduction="sum")
        )
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_epoch(model, loader, optimizer, device, beta=1.0):
    model.train()
    total_loss = 0.0

    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, _, _ = model.loss_function(data, recon, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader.dataset)

def extract_latent(model, loader, device):
    model.eval()
    latents, labels = [], []

    with torch.no_grad():
        for data, label in loader:
            mu, _ = model.encode(data.to(device))
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)
