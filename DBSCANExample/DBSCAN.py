import numpy as np
import matplotlib.pyplot as plt # Standard way to import pyplot
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons # For generating the synthetic dataset
from sklearn.metrics import silhouette_score # To objectively compare clustering results
import pandas as pd # Included for general data science context, often implicitly needed for scikit-learn helper functions

# Note: matplotlib.use('TkAgg') is often specific to certain environments.
# For a GitHub-friendly script, it's generally best practice to remove it
# unless there's a specific requirement for that backend, as Matplotlib
# can usually select an appropriate one for the environment.

def generate_moons_dataset(n_samples=1500, noise=0.05, random_state=42):
    """
    Generates a synthetic 'moons' dataset.
    This dataset is often used to demonstrate the strengths of density-based
    clustering algorithms like DBSCAN over centroid-based ones like K-Means
    when clusters have non-linear shapes.

    Args:
        n_samples (int): The total number of points generated.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_state (int): Determines random number generation for dataset creation.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The generated samples (features).
            - y (numpy.ndarray): The true class labels for each sample (used for comparison).
    """
    print(f"Generating 'make_moons' dataset with {n_samples} samples and noise={noise}...")
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    print(f"Generated data shape: X={X.shape}, y={y.shape}")
    return X, y

def apply_dbscan_clustering(X_data, eps=0.3, min_samples=5):
    """
    Applies DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    clustering to the dataset.

    DBSCAN groups together points that are closely packed together (points with many
    nearby neighbors), marking as outliers points that lie alone in low-density regions.

    Args:
        X_data (numpy.ndarray): The input feature data.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood
                           for a point to be considered as a core point.

    Returns:
        tuple: A tuple containing:
            - model_dbscan (sklearn.cluster.DBSCAN): The fitted DBSCAN model.
            - y_pred_dbscan (numpy.ndarray): The cluster labels for each sample.
                                            -1 indicates noise (outliers).
    """
    print(f"\nApplying DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
    model_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    model_dbscan.fit(X_data)
    y_pred_dbscan = model_dbscan.labels_.astype(int)
    n_clusters_dbscan = len(set(y_pred_dbscan)) - (1 if -1 in y_pred_dbscan else 0)
    n_noise_dbscan = list(y_pred_dbscan).count(-1)
    print(f"DBSCAN found {n_clusters_dbscan} clusters and {n_noise_dbscan} noise points.")

    # Evaluate using Silhouette Score (only if more than 1 cluster and no only noise points)
    if n_clusters_dbscan > 1 and n_clusters_dbscan + n_noise_dbscan < len(X_data):
        silhouette = silhouette_score(X_data, y_pred_dbscan)
        print(f"DBSCAN Silhouette Score: {silhouette:.4f}")
    else:
        print("DBSCAN Silhouette Score cannot be computed (less than 2 clusters or all noise).")

    return model_dbscan, y_pred_dbscan

def apply_kmeans_clustering(X_data, n_clusters=2, random_state=42):
    """
    Applies K-Means clustering to the dataset.

    K-Means is a centroid-based clustering algorithm that aims to partition
    n observations into k clusters, where each observation belongs to the cluster
    with the nearest mean (centroid).

    Args:
        X_data (numpy.ndarray): The input feature data.
        n_clusters (int): The number of clusters to form.
        random_state (int): Determines random number generation for centroid initialization.

    Returns:
        tuple: A tuple containing:
            - model_kmeans (sklearn.cluster.KMeans): The fitted K-Means model.
            - y_pred_kmeans (numpy.ndarray): The cluster labels for each sample.
    """
    print(f"\nApplying K-Means clustering (n_clusters={n_clusters})...")
    # n_init='auto' (default in recent scikit-learn) or an integer to specify how many
    # times the k-means algorithm is run with different centroid seeds.
    # random_state ensures reproducibility of centroid initialization.
    model_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model_kmeans.fit(X_data)
    y_pred_kmeans = model_kmeans.labels_.astype(int)
    print(f"K-Means found {n_clusters} clusters.")

    # Evaluate using Silhouette Score
    silhouette = silhouette_score(X_data, y_pred_kmeans)
    print(f"K-Means Silhouette Score: {silhouette:.4f}")

    return model_kmeans, y_pred_kmeans

def visualize_clustering_results(X_data, y_pred_dbscan, y_pred_kmeans, title1, title2):
    """
    Visualizes the clustering results from DBSCAN and K-Means side-by-side.

    Args:
        X_data (numpy.ndarray): The original feature data.
        y_pred_dbscan (numpy.ndarray): Cluster labels from DBSCAN.
        y_pred_kmeans (numpy.ndarray): Cluster labels from K-Means.
        title1 (str): Title for the first subplot (DBSCAN).
        title2 (str): Title for the second subplot (K-Means).
    """
    print("\n--- Visualizing Clustering Results ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Adjust figure size for better readability

    # Extract x and y coordinates for plotting
    x1 = X_data[:, 0]
    x2 = X_data[:, 1]

    # Draw the DBSCAN chart in the first subplot
    # 'cmap' sets the colormap. 's' sets marker size.
    axes[0].scatter(x1, x2, c=y_pred_dbscan, cmap='rainbow', s=10, alpha=0.8)
    axes[0].set_title(title1)
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Draw the KMeans chart in the second subplot
    axes[1].scatter(x1, x2, c=y_pred_kmeans, cmap='rainbow', s=10, alpha=0.8)
    axes[1].set_title(title2)
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.show()

if __name__ == "__main__":
    # Parameters for dataset generation
    N_SAMPLES = 1500
    NOISE_LEVEL = 0.05
    DATA_RANDOM_STATE = 42

    # Parameters for DBSCAN
    DBSCAN_EPS = 0.3
    DBSCAN_MIN_SAMPLES = 5

    # Parameters for K-Means
    KMEANS_N_CLUSTERS = 2
    KMEANS_RANDOM_STATE = 42

    # 1. Generate the 'make_moons' dataset
    X_data, y_true_labels = generate_moons_dataset(n_samples=N_SAMPLES, noise=NOISE_LEVEL, random_state=DATA_RANDOM_STATE)

    # 2. Apply DBSCAN Clustering
    dbscan_model, dbscan_predictions = apply_dbscan_clustering(X_data, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

    # 3. Apply K-Means Clustering
    kmeans_model, kmeans_predictions = apply_kmeans_clustering(X_data, n_clusters=KMEANS_N_CLUSTERS, random_state=KMEANS_RANDOM_STATE)

    # 4. Visualize and Compare Results
    visualize_clustering_results(X_data, dbscan_predictions, kmeans_predictions,
                                 "DBSCAN Clustering Results", "K-Means Clustering Results")

    print("\nScript execution complete.")
