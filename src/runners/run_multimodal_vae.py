import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import MusicDataset
from src.training.train_multimodal_vae import MultiModalVAE, train_epoch, extract_latent

def main():
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    config = {
        "audio_dim": 20,
        "lyrics_dim": 30,
        "hidden_dim": 128,
        "latent_dim": 16,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "beta": 1.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load features
    with open(DATA_DIR / "processed_features.pkl", "rb") as f:
        audio_data = pickle.load(f)

    with open(DATA_DIR / "lyrics_features.pkl", "rb") as f:
        lyrics_data = pickle.load(f)

    combined_features = np.concatenate(
        [audio_data["features"], lyrics_data["lyrics_features"]], axis=1
    )
    labels = audio_data["labels"]
    genre_mapping = audio_data["genre_mapping"]

    dataset = MusicDataset(combined_features, labels)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = MultiModalVAE(
        audio_dim=config["audio_dim"],
        lyrics_dim=config["lyrics_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_loss = float("inf")
    print("\nStarting Training...")

    for epoch in range(config["num_epochs"]):
        avg_loss = train_epoch(model, loader, optimizer, device, beta=config["beta"])

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": avg_loss, "config": config},
                RESULTS_DIR / "best_multimodal_vae.pth",
            )

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

    latent_features, latent_labels = extract_latent(model, loader, device)
    latent_data = {
        "latent_features": latent_features,
        "labels": latent_labels,
        "genre_mapping": genre_mapping,
        "config": config,
    }

    with open(DATA_DIR / "latent_features_multimodal.pkl", "wb") as f:
        pickle.dump(latent_data, f)

    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        RESULTS_DIR / "final_multimodal_vae.pth",
    )

    print("\nFiles saved:")
    print(" - results/best_multimodal_vae.pth")
    print(" - results/final_multimodal_vae.pth")
    print(" - data/latent_features_multimodal.pkl")

if __name__ == "__main__":
    main()
