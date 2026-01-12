import torch
import numpy as np


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = 0.0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = model.loss_function(recon, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader.dataset)


def extract_latent(model, loader, device):
    model.eval()
    Z, y = [], []

    with torch.no_grad():
        for x, labels in loader:
            mu, _ = model.encode(x.to(device))
            Z.append(mu.cpu().numpy())
            y.append(labels.numpy())

    return np.vstack(Z), np.concatenate(y)
