import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage # For hierarchical clustering visualization
import matplotlib.pyplot as plt # Standard way to import pyplot
import pandas as pd # Included for general data science context

# Note: matplotlib.use('TkAgg') is often specific to certain environments.
# For a GitHub-friendly script, it's generally best practice to remove it
# unless there's a specific requirement for that backend, as Matplotlib
# can usually select an appropriate one for the environment.

def define_custom_data():
    """
    Defines a small, custom 2D dataset for hierarchical clustering demonstration.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the data points.
    """
    print("Defining custom 2D dataset...")
    # Each row is a data point, each column is a feature.
    # Example: [[x1, y1], [x2, y2], ...]
    x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])
    print("Custom data points:\n", x)
    return x

def create_linkage_matrix(data, method='single'):
    """
    Creates the linkage matrix for hierarchical clustering.

    The linkage matrix encodes the hierarchical clustering structure.
    It shows which clusters were merged at each step, and the distance
    between the merged clusters.

    Args:
        data (numpy.ndarray): The input feature data.
        method (str): The linkage method to use.
                      'single': minimum distance between observations in different clusters.
                      'complete': maximum distance between observations in different clusters.
                      'average': average distance between observations in different clusters.
                      'ward': minimizes the variance of the clusters being merged (often good for general purpose).

    Returns:
        numpy.ndarray: The linkage matrix.
    """
    print(f"\nCreating linkage matrix using '{method}' method...")
    # `linkage` performs hierarchical clustering on a dataset `x` and returns
    # the hierarchical clustering encoded as a linkage matrix.
    # Each row of the linkage matrix has the format [idx1, idx2, distance, num_samples].
    linkage_matrix = linkage(data, method=method)
    print("Linkage Matrix (first 5 rows if available):\n", linkage_matrix[:5] if len(linkage_matrix) > 5 else linkage_matrix)
    return linkage_matrix

def visualize_hierarchical_clustering(data, linkage_matrix, title_scatter='Given Chart', title_dendrogram='Hierarchical Clustering Dendrogram'):
    """
    Visualizes the original data points and the hierarchical clustering dendrogram.

    Args:
        data (numpy.ndarray): The original data points.
        linkage_matrix (numpy.ndarray): The linkage matrix from hierarchical clustering.
        title_scatter (str): Title for the scatter plot.
        title_dendrogram (str): Title for the dendrogram plot.
    """
    print("\n--- Visualizing Original Data and Dendrogram ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Adjust figure size for better readability

    # Scatter plot of the original data points
    axes[0].scatter(data[:, 0], data[:, 1], s=100, alpha=0.8, edgecolors='w') # s=marker size, edgecolors for visibility
    axes[0].set_title(title_scatter)
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Dendrogram plot
    # The dendrogram visually represents the hierarchy of clusters.
    # Each merge is represented by a U-shaped line. The height of the U represents
    # the distance at which the clusters were merged.
    dendrogram(linkage_matrix, ax=axes[1],
               orientation='top', # 'top' or 'left'
               distance_sort='descending', # Sort leaves by distance
               show_leaf_counts=True) # Show number of original observations in each leaf cluster
    axes[1].set_title(title_dendrogram)
    axes[1].set_xlabel("Data Points / Clusters")
    axes[1].set_ylabel("Distance")
    axes[1].tick_params(axis='x', rotation=45) # Rotate x-axis labels if needed

    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.show()

if __name__ == "__main__":
    # 1. Define the custom 2D dataset
    custom_data = define_custom_data()

    # 2. Create the linkage matrix using 'single' method
    # 'single' linkage considers the minimum distance between points in two clusters.
    # This often leads to "chaining" behavior.
    linkage_matrix_single = create_linkage_matrix(custom_data, method='single')

    # 3. Visualize the original data and the generated dendrogram
    visualize_hierarchical_clustering(custom_data, linkage_matrix_single)

    print("\nScript execution complete.")
