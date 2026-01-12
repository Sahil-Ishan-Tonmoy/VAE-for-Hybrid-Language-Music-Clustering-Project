import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MusicClusterer:
    """
    Clustering methods for music features
    """
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.cluster_labels = None
        self.clusterer = None
    
    def fit_kmeans(self, features):
        """K-Means clustering"""
        print(f"Performing K-Means clustering (k={self.n_clusters})...")
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.clusterer.fit_predict(features)
        return self.cluster_labels
    
    def fit_agglomerative(self, features):
        """Agglomerative (Hierarchical) clustering"""
        print(f"Performing Agglomerative clustering (k={self.n_clusters})...")
        self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.cluster_labels = self.clusterer.fit_predict(features)
        return self.cluster_labels
    
    def fit_dbscan(self, features, eps=0.5, min_samples=5):
        """DBSCAN clustering"""
        print(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = self.clusterer.fit_predict(features)
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        print(f"DBSCAN found {n_clusters} clusters")
        return self.cluster_labels


def apply_pca(features, n_components=16):
    """Apply PCA for dimensionality reduction (baseline)"""
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(features)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA explained variance: {explained_var:.4f}")
    return pca_features, pca


def visualize_2d(features, labels, true_labels, genre_mapping, 
                 method='tsne', save_path='results/visualization.png'):
    """
    Visualize high-dimensional features in 2D
    
    Args:
        features: High-dimensional features
        labels: Cluster labels
        true_labels: True genre labels
        genre_mapping: Genre name mapping
        method: 'tsne' or 'umap'
        save_path: Path to save plot
    """
    print(f"Creating 2D visualization using {method.upper()}...")
    
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = reducer.fit_transform(features)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create inverse genre mapping
    inv_genre_mapping = {v: k for k, v in genre_mapping.items()}
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Predicted clusters
    scatter1 = axes[0].scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=0.6,
        s=20
    )
    axes[0].set_title(f'Predicted Clusters ({method.upper()})', fontsize=14)
    axes[0].set_xlabel('Component 1', fontsize=12)
    axes[0].set_ylabel('Component 2', fontsize=12)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: True genres
    scatter2 = axes[1].scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=true_labels, 
        cmap='tab10', 
        alpha=0.6,
        s=20
    )
    axes[1].set_title(f'True Genres ({method.upper()})', fontsize=14)
    axes[1].set_xlabel('Component 1', fontsize=12)
    axes[1].set_ylabel('Component 2', fontsize=12)
    cbar = plt.colorbar(scatter2, ax=axes[1], label='Genre')
    
    # Add genre names to colorbar
    genre_names = [inv_genre_mapping[i] for i in range(len(genre_mapping))]
    cbar.set_ticks(range(len(genre_names)))
    cbar.set_ticklabels(genre_names, fontsize=8)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def plot_cluster_distribution(cluster_labels, true_labels, genre_mapping, 
                               save_path='results/cluster_distribution.png'):
    """
    Plot distribution of true genres across predicted clusters
    """
    print("Creating cluster distribution plot...")
    
    # Create inverse genre mapping
    inv_genre_mapping = {v: k for k, v in genre_mapping.items()}
    genre_names = [inv_genre_mapping[i] for i in range(len(genre_mapping))]
    
    # Create confusion matrix
    n_clusters = len(set(cluster_labels))
    confusion = np.zeros((n_clusters, len(genre_mapping)))
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        for genre_id in range(len(genre_mapping)):
            genre_mask = true_labels == genre_id
            confusion[cluster_id, genre_id] = np.sum(cluster_mask & genre_mask)
    
    # Normalize by row (cluster)
    confusion_normalized = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        confusion_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='YlOrRd',
        xticklabels=genre_names,
        yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
        cbar_kws={'label': 'Proportion'}
    )
    plt.title('Genre Distribution Across Clusters', fontsize=14)
    plt.xlabel('True Genre', fontsize=12)
    plt.ylabel('Predicted Cluster', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cluster distribution saved to {save_path}")
    plt.close()


def compare_methods(features, true_labels, n_clusters=10):
    """
    Compare different clustering methods
    
    Returns:
        results: Dictionary with cluster labels for each method
    """
    results = {}
    
    # K-Means on features
    clusterer = MusicClusterer(n_clusters=n_clusters)
    results['kmeans'] = clusterer.fit_kmeans(features)
    
    # Agglomerative on features
    clusterer = MusicClusterer(n_clusters=n_clusters)
    results['agglomerative'] = clusterer.fit_agglomerative(features)
    
    # DBSCAN on features (auto-detect clusters)
    clusterer = MusicClusterer(n_clusters=n_clusters)
    results['dbscan'] = clusterer.fit_dbscan(features, eps=1.5, min_samples=3)
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Clustering Module Test")
    print("=" * 70)
    
    # Create dummy data
    n_samples = 100
    n_features = 16
    n_clusters = 10
    
    dummy_features = np.random.randn(n_samples, n_features)
    dummy_labels = np.random.randint(0, n_clusters, n_samples)
    
    # Test K-Means
    clusterer = MusicClusterer(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_kmeans(dummy_features)
    
    print(f"\nCluster labels shape: {cluster_labels.shape}")
    print(f"Number of unique clusters: {len(set(cluster_labels))}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    print("\nClustering module test successful!")