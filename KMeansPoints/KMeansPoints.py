import numpy as np
import matplotlib.pyplot as plt # Standard way to import pyplot
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs # For generating the synthetic dataset
from sklearn.metrics import silhouette_score # For objective evaluation
import pandas as pd # Included for general data science context

# Note: matplotlib.use('TkAgg') is often specific to certain environments.
# For a GitHub-friendly script, it's generally best practice to remove it
# unless there's a specific requirement for that backend, as Matplotlib
# can usually select an appropriate one for the environment.

def generate_blobs_dataset(n_samples=100, n_centers=5, cluster_std=2, random_state=42):
    """
    Generates a synthetic 'blobs' dataset.
    This dataset is commonly used to demonstrate centroid-based clustering algorithms
    like K-Means, as it consists of isotropic Gaussian blobs (spherical clusters).

    Args:
        n_samples (int): The total number of points generated.
        n_centers (int): The number of centers (clusters) to generate.
        cluster_std (float): The standard deviation of the clusters.
        random_state (int): Determines random number generation for dataset creation.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The generated samples (features).
            - y_true (numpy.ndarray): The true cluster labels for each sample (used for comparison/understanding).
    """
    print(f"Generating 'make_blobs' dataset with {n_samples} samples, {n_centers} centers, and cluster_std={cluster_std}...")
    X, y_true = make_blobs(n_samples=n_samples, centers=n_centers, cluster_std=cluster_std, random_state=random_state)
    print(f"Generated data shape: X={X.shape}, y_true={y_true.shape}")
    print(f"True cluster labels (first 10): {y_true[:10]}")
    return X, y_true

def apply_kmeans_clustering(X_data, n_clusters, random_state=42):
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
            - y_predicted (numpy.ndarray): The cluster labels for each sample predicted by K-Means.
    """
    print(f"\nApplying K-Means clustering with {n_clusters} clusters...")
    # n_init='auto' (default in recent scikit-learn) or an integer to specify how many
    # times the k-means algorithm is run with different centroid seeds.
    # random_state ensures reproducibility of centroid initialization.
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_data) # Train the model on the data
    y_predicted = model.predict(X_data) # Get the cluster labels for each data point

    print(f"Predicted cluster labels (first 10): {y_predicted[:10]}")

    # Evaluate using Silhouette Score
    silhouette = silhouette_score(X_data, y_predicted)
    print(f"K-Means Silhouette Score: {silhouette:.4f}")

    return model, y_predicted

def visualize_clustering_results(X_data, y_predicted, title="K-Means Clustering on Blobs"):
    """
    Visualizes the K-Means clustering results.

    Args:
        X_data (numpy.ndarray): The original feature data.
        y_predicted (numpy.ndarray): The cluster labels predicted by K-Means.
        title (str): The title for the plot.
    """
    print("\n--- Visualizing K-Means Clustering Results ---")
    plt.figure(figsize=(8, 7)) # Adjust figure size for better readability
    # Scatter plot of the data points, colored by their predicted cluster label.
    # `cmap='rainbow'` applies a colormap to differentiate clusters.
    # `s=50` sets the marker size.
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_predicted, cmap='rainbow', s=50, alpha=0.8, edgecolors='w')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    # Parameters for dataset generation
    N_SAMPLES = 300 # Increased samples for better visualization
    N_CENTERS = 5
    CLUSTER_STD = 2.0
    DATA_RANDOM_STATE = 42

    # Parameters for K-Means clustering
    KMEANS_N_CLUSTERS = 5 # Should match N_CENTERS for 'make_blobs' to find true clusters
    KMEANS_RANDOM_STATE = 42

    # 1. Generate the 'make_blobs' dataset
    X_data, y_true_labels = generate_blobs_dataset(
        n_samples=N_SAMPLES,
        n_centers=N_CENTERS,
        cluster_std=CLUSTER_STD,
        random_state=DATA_RANDOM_STATE
    )

    # 2. Apply K-Means Clustering
    kmeans_model, kmeans_predictions = apply_kmeans_clustering(
        X_data,
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE
    )

    # 3. Visualize the Clustering Results
    visualize_clustering_results(X_data, kmeans_predictions)

    print("\nScript execution complete.")
