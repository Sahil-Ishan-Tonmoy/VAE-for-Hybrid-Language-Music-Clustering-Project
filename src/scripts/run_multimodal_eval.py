from feature_loader import load_features
from comparison import kmeans_evaluate, compare_two_methods
import pickle

def main():
    print("="*70)
    print("Multi-Modal VAE Evaluation")
    print("="*70)

    mm_features, labels = load_features(
        "data/latent_features_multimodal.pkl"
    )
    audio_features, _ = load_features(
        "data/latent_features.pkl"
    )

    mm_metrics = kmeans_evaluate(mm_features, labels)
    audio_metrics = kmeans_evaluate(audio_features, labels)

    comparison = compare_two_methods(
        "Multi-Modal", mm_metrics,
        "Audio-Only", audio_metrics
    )

    for metric, values in comparison.items():
        print(f"{metric:<25} {values['diff_pct']:>8.2f}%")

    with open("results/multimodal_comparison.pkl", "wb") as f:
        pickle.dump(comparison, f)


if __name__ == "__main__":
    main()
