import torch
import numpy as np


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, _, _ = model.loss_function(x, recon, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader.dataset)


def extract_latent(model, features, device):
    model.eval()
    with torch.no_grad():
        return model.get_latent_features(
            torch.tensor(features).float().to(device)
        ).cpu().numpy()
