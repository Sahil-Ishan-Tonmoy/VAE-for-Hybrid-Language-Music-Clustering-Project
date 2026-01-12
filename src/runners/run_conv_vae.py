import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.models.conv_vae import ConvVAE
from src.training.train_conv_vae import train_epoch, extract_latent


def main():
    DATA_DIR = Path("data")

    with open(DATA_DIR / "processed_features.pkl", "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    labels = data["labels"]

    loader = DataLoader(
        TensorDataset(torch.tensor(features).float()),
        batch_size=32,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(input_dim=20, latent_dim=16, beta=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        train_epoch(model, loader, optimizer, device)

    Z = extract_latent(model, features, device)

    with open(DATA_DIR / "latent_features_conv_vae.pkl", "wb") as f:
        pickle.dump({"features": Z, "labels": labels}, f)


if __name__ == "__main__":
    main()
