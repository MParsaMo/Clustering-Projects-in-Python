import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage # For hierarchical clustering visualization
import matplotlib.pyplot as plt # Standard way to import pyplot
from sklearn.cluster import AgglomerativeClustering # For cutting the dendrogram to form clusters
import os # For checking file existence and creating dummy data

# Note: matplotlib.use('TkAgg') is often specific to certain environments.
# For a GitHub-friendly script, it's generally best practice to remove it
# unless there's a specific requirement for that backend, as Matplotlib
# can usually select an appropriate one for the environment.

# Define the file path for the dataset
CSV_FILE_PATH = 'shopping_data.csv'

def load_shopping_data(file_path):
    """
    Loads shopping data from a CSV file.
    If the file is not found, a dummy CSV is created for demonstration purposes.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'shopping_data.csv' is in the same directory as the script.")
        print("Creating a dummy 'shopping_data.csv' for demonstration purposes.")
        # Create a dummy CSV for demonstration
        dummy_data = {
            'CustomerID': range(1, 21),
            'Gender': ['Male', 'Female'] * 10,
            'Age': np.random.randint(18, 60, 20),
            'Annual Income (k$)': np.random.randint(15, 120, 20),
            'Spending Score (1-100)': np.random.randint(1, 100, 20)
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)
        print("Dummy 'shopping_data.csv' created. Please replace it with your actual data.")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def prepare_features(dataframe, feature_indices):
    """
    Selects specific columns from the DataFrame and converts them to a NumPy array.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        feature_indices (list): A list of integer indices for the columns to select.

    Returns:
        numpy.ndarray: A 2D NumPy array containing the selected features.
    """
    print(f"\n--- Preparing Features (Columns at indices {feature_indices}) ---")
    # iloc[:, 3:5] selects columns with index 3 and 4 (4th and 5th columns)
    # .values converts the Pandas DataFrame slice to a NumPy array.
    data_features = dataframe.iloc[:, feature_indices].values
    print(f"Selected features shape: {data_features.shape}")
    print(f"First 5 rows of selected features:\n{data_features[:5]}")
    return data_features

def create_linkage_matrix(data, method='ward'):
    """
    Creates the linkage matrix for hierarchical clustering.

    Args:
        data (numpy.ndarray): The input feature data.
        method (str): The linkage method to use. 'ward' minimizes the variance
                      of the clusters being merged, often suitable for general purpose.

    Returns:
        numpy.ndarray: The linkage matrix.
    """
    print(f"\nCreating linkage matrix using '{method}' method...")
    # `linkage` performs hierarchical clustering and returns the hierarchical clustering
    # encoded as a linkage matrix.
    linkage_matrix = linkage(data, method=method)
    print("Linkage Matrix (first 5 rows if available):\n", linkage_matrix[:5] if len(linkage_matrix) > 5 else linkage_matrix)
    return linkage_matrix

def visualize_dendrogram_and_clustering(data, linkage_matrix, n_clusters=5):
    """
    Visualizes the hierarchical clustering dendrogram and the resulting
    AgglomerativeClustering plot side-by-side.

    Args:
        data (numpy.ndarray): The original feature data.
        linkage_matrix (numpy.ndarray): The linkage matrix from hierarchical clustering.
        n_clusters (int): The number of clusters to form for AgglomerativeClustering.
                          This number is typically chosen by observing the dendrogram.
    """
    print(f"\n--- Visualizing Dendrogram and Agglomerative Clustering (n_clusters={n_clusters}) ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Adjust figure size for better readability

    # Dendrogram plot
    # The dendrogram visually represents the hierarchy of clusters.
    # It helps in determining the optimal number of clusters by looking for the
    # largest vertical distance that does not cross any horizontal bar.
    dendrogram(linkage_matrix, ax=axes[0],
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    axes[0].set_title('Shopping Data Dendrogram')
    axes[0].set_xlabel('Data Points / Clusters')
    axes[0].set_ylabel('Distance')
    axes[0].tick_params(axis='x', rotation=45) # Rotate x-axis labels if needed

    # Creating the AgglomerativeClustering model
    # AgglomerativeClustering performs hierarchical clustering using a bottom-up approach.
    # n_clusters: The number of clusters to find.
    # linkage: The linkage criterion to use (e.g., 'ward', 'complete', 'average', 'single').
    #          'ward' is often used as it minimizes the variance of the clusters being merged.
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    # fit_predict fits the model and returns the cluster labels for each sample
    agglomerative_labels = cluster_model.fit_predict(data)
    print(f"Agglomerative Clustering found {len(np.unique(agglomerative_labels))} clusters.")

    # AgglomerativeClustering plot
    axes[1].scatter(data[:, 0], data[:, 1], c=agglomerative_labels, cmap='rainbow', s=50, alpha=0.8, edgecolors='w')
    axes[1].set_title('Market Segmentation by Agglomerative Clustering')
    axes[1].set_xlabel('Annual Income (k$)')
    axes[1].set_ylabel('Spending Score (1-100)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.show()

if __name__ == "__main__":
    # Define parameters
    FEATURE_COL_INDICES = [3, 4] # Columns for 'Annual Income (k$)' and 'Spending Score (1-100)'
    LINKAGE_METHOD = 'ward'
    NUM_CLUSTERS_FOR_AGGLOMERATIVE = 5 # This is an example, best K is usually determined from dendrogram

    # 1. Load the shopping data
    shopping_df = load_shopping_data(CSV_FILE_PATH)
    if shopping_df is None:
        exit() # Exit if data loading failed

    # 2. Prepare features (select specific columns)
    features_for_clustering = prepare_features(shopping_df, FEATURE_COL_INDICES)

    # 3. Create the linkage matrix
    # 'ward' linkage is suitable for this type of data to form compact, spherical clusters.
    linkage_matrix = create_linkage_matrix(features_for_clustering, method=LINKAGE_METHOD)

    # 4. Visualize the dendrogram and the Agglomerative Clustering results
    visualize_dendrogram_and_clustering(features_for_clustering, linkage_matrix,
                                        n_clusters=NUM_CLUSTERS_FOR_AGGLOMERATIVE)

    print("\nScript execution complete.")
