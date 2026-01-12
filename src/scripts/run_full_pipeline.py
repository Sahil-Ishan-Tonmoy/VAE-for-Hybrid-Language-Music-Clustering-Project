from feature_loader import load_features
from clustering import MusicClusterer, apply_pca
from evaluation import evaluate_clustering, save_metrics_table

def main():
    features, labels = load_features("data/latent_features.pkl")

    clusterer = MusicClusterer(n_clusters=10)
    kmeans_labels = clusterer.fit_kmeans(features)

    metrics = evaluate_clustering(features, kmeans_labels, labels)
    save_metrics_table(
        {"VAE + KMeans": metrics},
        "results/clustering_metrics.csv"
    )


if __name__ == "__main__":
    main()
