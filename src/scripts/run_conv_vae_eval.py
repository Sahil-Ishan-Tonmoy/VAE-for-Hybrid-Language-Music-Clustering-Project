from feature_loader import load_features
from comparison import kmeans_evaluate, compare_two_methods
import pickle

def main():
    print("="*70)
    print("Conv VAE Clustering Evaluation")
    print("="*70)

    conv_features, labels = load_features(
        "data/latent_features_conv_vae.pkl"
    )

    conv_metrics = kmeans_evaluate(conv_features, labels)

    print("\nConv VAE Metrics:")
    for k, v in conv_metrics.items():
        print(f"{k:.<40} {v:.4f}")

    try:
        vae_features, vae_labels = load_features(
            "data/latent_features.pkl"
        )
        vae_metrics = kmeans_evaluate(vae_features, vae_labels)

        comparison = compare_two_methods(
            "Conv VAE", conv_metrics,
            "VAE", vae_metrics
        )

        print("\nConv VAE vs VAE:")
        for k, v in comparison.items():
            print(f"{k:<25} {v['diff_pct']:>8.2f}%")

    except FileNotFoundError:
        print("Standard VAE not found")

    with open("results/conv_vae_results.pkl", "wb") as f:
        pickle.dump(conv_metrics, f)


if __name__ == "__main__":
    main()
