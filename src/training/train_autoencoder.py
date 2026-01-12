# src/training/train_autoencoder.py
import torch
import torch.nn.functional as F
import numpy as np

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = F.mse_loss(recon, x, reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader.dataset)


def extract_latent(model, loader, device):
    model.eval()
    Z, y = [], []

    with torch.no_grad():
        for x, labels in loader:
            z = model.encode(x.to(device))
            Z.append(z.cpu().numpy())
            y.append(labels.numpy())

    return np.vstack(Z), np.concatenate(y)
