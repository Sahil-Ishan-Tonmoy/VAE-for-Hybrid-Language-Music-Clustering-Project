import torch
import pickle
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.vae import MusicVAE
from src.dataset import MusicDataset
from src.training.train_beta_vae import train_epoch, extract_latent


def main():
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")

    with open(DATA_DIR / "processed_features.pkl", "rb") as f:
        data = pickle.load(f)

    loader = DataLoader(
        MusicDataset(data["features"], data["labels"]),
        batch_size=32,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for beta in [1.0, 2.0, 4.0]:
        model = MusicVAE(input_dim=20, latent_dim=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(50):
            loss = train_epoch(model, loader, optimizer, device, beta)

        Z, y = extract_latent(model, loader, device)

        with open(DATA_DIR / f"latent_features_beta{beta}.pkl", "wb") as f:
            pickle.dump({"features": Z, "labels": y}, f)


if __name__ == "__main__":
    main()
