🤖 Clustering Projects in Python
This repository contains several clustering examples using popular machine learning and NLP techniques in Python. The goal is to showcase different ways to discover patterns and groupings in both structured and unstructured data using algorithms like:

K-Means

DBSCAN

Hierarchical Clustering

Agglomerative Clustering

K-Means for Text Clustering with TF-IDF

🧰 Requirements
Install the required libraries:
pip install numpy pandas matplotlib scikit-learn scipy nltk


If you're working with the NLP example:
import nltk
nltk.download('punkt')
nltk.download('stopwords')

📁 Project List
1. 🌙 Moons Dataset: DBSCAN vs KMeans
File: moons_dbscan_vs_kmeans.py
Goal: Compare DBSCAN and KMeans on a two-moon shaped dataset.

Generates a noisy dataset with make_moons()

Applies both DBSCAN and KMeans

Visualizes clustering results side-by-side


2. 🌳 Hierarchical Clustering with Dendrogram (Toy Data)
File: dendrogram_example.py
Goal: Illustrate how linkage-based hierarchical clustering works.

Manually creates a small dataset

Plots both the data points and a dendrogram

Uses scipy.cluster.hierarchy.linkage() and dendrogram()

3. 🛍️ Market Segmentation: Agglomerative Clustering
File: market_segmentation.py
Goal: Cluster customers based on income and spending score.

Loads shopping_data.csv

Uses a dendrogram to decide the number of clusters

Applies AgglomerativeClustering for final segmentation

Visualizes both the dendrogram and segmentation

4. 🔵 KMeans on Synthetic Blobs
File: kmeans_blobs.py
Goal: Cluster synthetic blobs with 5 true centers using KMeans.

Uses make_blobs() to generate clusters

Fits and predicts with KMeans(n_clusters=5)

Plots final clustered result

5. 🧠 Text Clustering with TF-IDF + KMeans
File: text_kmeans_clustering.py
Goal: Cluster related sentences using TF-IDF + KMeans.

Tokenizes and stems text using nltk

Converts text to vectors using TfidfVectorizer

Clusters using KMeans

Outputs grouped sentences per cluster

Example Output:
CLUSTER  0 :
    SENTENCE  0 :  Investing in stocks and trading with them are not that easy
    SENTENCE  1 :  Warren Buffet is famous for making good investments...

CLUSTER  1 :
    SENTENCE  0 :  Quantum physics is quite important...
    SENTENCE  1 :  Software engineering is hotter and hotter...

🧪 Summary of Techniques
| Technique                  | Algorithm          | Dataset Used              | Purpose                    |
| -------------------------- | ------------------ | ------------------------- | -------------------------- |
| DBSCAN vs KMeans           | DBSCAN, KMeans     | `make_moons()`            | Compare clustering shapes  |
| Hierarchical Clustering    | Linkage/Dendrogram | Manual toy data           | Visual explanation         |
| Agglomerative Clustering   | Agglomerative      | `shopping_data.csv`       | Market segmentation        |
| KMeans on blobs            | KMeans             | `make_blobs()`            | Clustering demo            |
| KMeans for Text Clustering | TF-IDF + KMeans    | List of example sentences | Grouping similar sentences |

📌 Notes
All examples use unsupervised learning — no labels are used to guide training.

KMeans works well when clusters are spherical and balanced.

DBSCAN can capture more complex cluster shapes but may label outliers as noise.

Hierarchical clustering helps understand structure and choose the number of clusters.

Text clustering is a powerful way to group documents without predefined categories.

📂 Dataset Info
shopping_data.csv must contain columns:
Annual Income (k$) and Spending Score (1-100) (columns 4 and 5).

📊 Visualizations
Each script (except text clustering) includes visual outputs like scatter plots or dendrograms using matplotlib.

👨‍💻 Author
Educational clustering projects to showcase different approaches in both structured and unstructured data using Python & Scikit-learn.
