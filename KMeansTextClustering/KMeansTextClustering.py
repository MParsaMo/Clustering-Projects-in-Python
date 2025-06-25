import collections
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np # For potential use in scikit-learn internals
import sys # For graceful exit if NLTK downloads fail

def initialize_nltk_resources():
    """
    Initializes NLTK resources by downloading necessary data if not already present.
    This ensures the tokenizer and stopwords are available.
    """
    print("Checking NLTK resources...")
    required_nltk_data = ['stopwords', 'punkt']
    for data_id in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data_id}' if data_id == 'punkt' else f'corpora/{data_id}')
            print(f"'{data_id}' NLTK data already present.")
        except nltk.downloader.DownloadError:
            print(f"Downloading NLTK '{data_id}' data...")
            try:
                nltk.download(data_id)
                print(f"'{data_id}' download complete.")
            except Exception as e:
                print(f"Error downloading '{data_id}': {e}. Please check your internet connection or run 'python -m nltk.downloader {data_id}' manually.")
                sys.exit(1) # Exit if essential data cannot be downloaded
    print("NLTK resources ready.")


def custom_tokenizer(text):
    """
    Custom tokenizer function that performs:
    1. Tokenization: Breaks text into individual words/tokens.
    2. Lowercasing: Converts all tokens to lowercase.
    3. Stop word removal: Filters out common, less meaningful words (e.g., 'the', 'is').
    4. Stemming: Reduces words to their base or root form (e.g., 'running' -> 'run').

    This preprocessing helps in reducing dimensionality and focusing on core terms.

    Args:
        text (str): The input sentence string.

    Returns:
        list: A list of preprocessed (stemmed, stop-word-removed) tokens.
    """
    # 1. Tokenize the text
    tokens = word_tokenize(text)

    # 2. Initialize Porter Stemmer
    stemmer = PorterStemmer()

    # 3. Get English stop words and convert to a set for faster lookup
    stop_words = set(stopwords.words('english'))

    # 4. Remove stopwords and stem tokens
    filtered_tokens = []
    for token in tokens:
        token_lower = token.lower()  # Convert to lowercase
        if token_lower.isalpha() and token_lower not in stop_words: # Check if alpha and not a stopword
            stemmed_token = stemmer.stem(token_lower) # Apply stemming
            filtered_tokens.append(stemmed_token)

    return filtered_tokens

def cluster_sentences(sentences_list, nb_of_clusters=2, random_state=42):
    """
    Clusters a list of sentences using TF-IDF vectorization and K-Means clustering.

    Args:
        sentences_list (list): A list of strings, where each string is a sentence.
        nb_of_clusters (int): The desired number of clusters.
        random_state (int): Seed for reproducibility of K-Means centroid initialization.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists
              of original sentence indices belonging to that cluster.
    """
    print(f"\n--- Clustering Sentences into {nb_of_clusters} Clusters ---")

    # Initialize TfidfVectorizer with the custom tokenizer.
    # The tokenizer handles lowercasing, stopword removal, and stemming.
    # analyzer='word' means features are single words.
    tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, analyzer='word')

    # Build a TF-IDF matrix for the sentences
    # fit_transform learns the vocabulary and IDF weights, then transforms sentences.
    print("Building TF-IDF matrix...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences_list)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (sentences x unique_terms)")

    # Initialize KMeans clustering model
    # n_init: Number of times the K-Means algorithm will be run with different centroid seeds.
    #         The final results will be the best output of n_init consecutive runs.
    # random_state: Ensures reproducibility of the centroid initialization.
    kmeans = KMeans(n_clusters=nb_of_clusters, n_init=10, random_state=random_state)

    # Fit K-Means to the TF-IDF matrix
    print("Fitting K-Means model...")
    kmeans.fit(tfidf_matrix)

    # Create a dictionary to store sentences by their cluster label
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i) # Store original sentence index

    print("Clustering complete.")
    return dict(clusters)

def display_clusters(clusters_dict, original_sentences):
    """
    Prints the sentences organized by their assigned clusters.

    Args:
        clusters_dict (dict): Dictionary of clusters (label: list of sentence indices).
        original_sentences (list): The list of original sentences.
    """
    print("\n--- Clustering Results ---")
    # Sort cluster labels for consistent output order
    sorted_cluster_labels = sorted(clusters_dict.keys())

    for cluster_label in sorted_cluster_labels:
        print(f"\nCLUSTER {cluster_label}:")
        if not clusters_dict[cluster_label]:
            print("\t(No sentences in this cluster)")
            continue
        for i, sentence_index in enumerate(clusters_dict[cluster_label]):
            print(f"\tSENTENCE {i+1}: {original_sentences[sentence_index]}")

if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    initialize_nltk_resources()

    # Define the sentences to be clustered
    sample_sentences = [
        "Quantum physics is quite important in science nowadays.",
        "Software engineering is hotter and hotter topic in the silicon valley",
        "Investing in stocks and trading with them are not that easy",
        "FOREX is the stock market for trading currencies",
        "Warren Buffet is famous for making good investments. He knows stock markets"
    ]

    # Determine the number of clusters (e.g., 2)
    NUM_CLUSTERS = 2
    KMEANS_RANDOM_STATE = 42 # For reproducibility of K-Means

    # Perform sentence clustering
    clustered_results = cluster_sentences(sample_sentences,
                                          nb_of_clusters=NUM_CLUSTERS,
                                          random_state=KMEANS_RANDOM_STATE)

    # Display the clustering results
    display_clusters(clustered_results, sample_sentences)

    print("\nScript execution complete.")
