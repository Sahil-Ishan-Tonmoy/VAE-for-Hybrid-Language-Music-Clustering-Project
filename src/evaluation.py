import numpy as np
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import pandas as pd


def compute_silhouette_score(features, labels):
    """
    Compute Silhouette Score
    Range: [-1, 1], higher is better
    """
    if len(set(labels)) <= 1:
        return 0.0
    score = silhouette_score(features, labels)
    return score


def compute_calinski_harabasz_score(features, labels):
    """
    Compute Calinski-Harabasz Index
    Higher is better
    """
    if len(set(labels)) <= 1:
        return 0.0
    score = calinski_harabasz_score(features, labels)
    return score


def compute_davies_bouldin_score(features, labels):
    """
    Compute Davies-Bouldin Index
    Lower is better
    """
    if len(set(labels)) <= 1:
        return float('inf')
    score = davies_bouldin_score(features, labels)
    return score


def compute_adjusted_rand_score(true_labels, pred_labels):
    """
    Compute Adjusted Rand Index
    Range: [-1, 1], higher is better (1 = perfect match)
    """
    score = adjusted_rand_score(true_labels, pred_labels)
    return score


def compute_nmi_score(true_labels, pred_labels):
    """
    Compute Normalized Mutual Information
    Range: [0, 1], higher is better
    """
    score = normalized_mutual_info_score(true_labels, pred_labels)
    return score


def compute_cluster_purity(true_labels, pred_labels):
    """
    Compute Cluster Purity
    Range: [0, 1], higher is better
    """
    n = len(true_labels)
    clusters = set(pred_labels)
    
    correct = 0
    for cluster in clusters:
        cluster_mask = pred_labels == cluster
        if np.sum(cluster_mask) == 0:
            continue
        # Find most common true label in this cluster
        true_labels_in_cluster = true_labels[cluster_mask]
        most_common = np.bincount(true_labels_in_cluster).argmax()
        correct += np.sum(true_labels_in_cluster == most_common)
    
    purity = correct / n
    return purity


def evaluate_clustering(features, cluster_labels, true_labels=None):
    """
    Compute all clustering metrics
    
    Args:
        features: Feature matrix
        cluster_labels: Predicted cluster labels
        true_labels: True labels (optional, for supervised metrics)
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # Unsupervised metrics (no ground truth needed)
    metrics['silhouette_score'] = compute_silhouette_score(features, cluster_labels)
    metrics['calinski_harabasz_score'] = compute_calinski_harabasz_score(features, cluster_labels)
    metrics['davies_bouldin_score'] = compute_davies_bouldin_score(features, cluster_labels)
    
    # Supervised metrics (require ground truth)
    if true_labels is not None:
        metrics['adjusted_rand_score'] = compute_adjusted_rand_score(true_labels, cluster_labels)
        metrics['nmi_score'] = compute_nmi_score(true_labels, cluster_labels)
        metrics['cluster_purity'] = compute_cluster_purity(true_labels, cluster_labels)
    
    return metrics


def print_metrics(metrics, method_name="Clustering"):
    """
    Print metrics in a formatted way
    """
    print(f"\n{'='*60}")
    print(f"{method_name} Metrics:")
    print(f"{'='*60}")
    
    for metric_name, value in metrics.items():
        # Format metric name
        formatted_name = metric_name.replace('_', ' ').title()
        
        # Determine if higher/lower is better
        if 'davies' in metric_name.lower():
            indicator = "↓ (lower is better)"
        else:
            indicator = "↑ (higher is better)"
        
        print(f"{formatted_name:.<40} {value:>8.4f}  {indicator}")
    
    print(f"{'='*60}")


def create_metrics_table(results_dict):
    """
    Create a comparison table of metrics across different methods
    
    Args:
        results_dict: Dict of {method_name: metrics_dict}
        
    Returns:
        DataFrame with metrics comparison
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df


def save_metrics_table(results_dict, save_path='results/clustering_metrics.csv'):
    """
    Save metrics comparison table to CSV
    """
    df = create_metrics_table(results_dict)
    df.to_csv(save_path)
    print(f"\nMetrics table saved to {save_path}")
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("Evaluation Metrics Module Test")
    print("=" * 70)
    
    # Create dummy data
    n_samples = 100
    n_features = 16
    n_clusters = 10
    
    dummy_features = np.random.randn(n_samples, n_features)
    dummy_cluster_labels = np.random.randint(0, n_clusters, n_samples)
    dummy_true_labels = np.random.randint(0, n_clusters, n_samples)
    
    # Compute metrics
    metrics = evaluate_clustering(dummy_features, dummy_cluster_labels, dummy_true_labels)
    
    # Print metrics
    print_metrics(metrics, "Test Clustering")
    
    print("\nEvaluation module test successful!")