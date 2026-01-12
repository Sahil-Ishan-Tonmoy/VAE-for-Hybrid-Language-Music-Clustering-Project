# src/comparison.py
from sklearn.cluster import KMeans
from evaluation import evaluate_clustering


def kmeans_evaluate(features, labels, n_clusters=10):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = model.fit_predict(features)
    return evaluate_clustering(features, cluster_labels, labels)


def compare_two_methods(name_a, metrics_a, name_b, metrics_b):
    comparison = {}

    for key in metrics_a:
        comparison[key] = {
            name_a: metrics_a[key],
            name_b: metrics_b[key],
            "diff_pct": (
                (metrics_a[key] - metrics_b[key]) / metrics_b[key] * 100
                if metrics_b[key] != 0 else 0
            )
        }

    return comparison
