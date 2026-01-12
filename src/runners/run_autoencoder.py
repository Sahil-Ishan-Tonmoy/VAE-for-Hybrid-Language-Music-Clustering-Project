import torch
import pickle
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.autoencoder import AutoEncoder
from src.dataset import MusicDataset
from src.training.train_autoencoder import train_epoch, extract_latent


def main():
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "processed_features.pkl", "rb") as f:
        data = pickle.load(f)

    dataset = MusicDataset(data["features"], data["labels"])
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(input_dim=20, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        loss = train_epoch(model, loader, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    Z, y = extract_latent(model, loader, device)

    with open(DATA_DIR / "latent_features_autoencoder.pkl", "wb") as f:
        pickle.dump({"features": Z, "labels": y}, f)


if __name__ == "__main__":
    main()
