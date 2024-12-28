import nltk
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the article text
def load_text(file_path):
    """Loads the text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# Path to your text file
file_path = "nyt_psilocybin.txt"
article_text = load_text(file_path)

# Download NLTK resources (only needed once)
nltk.download("punkt")

# Split the article into sentences
sentences = nltk.sent_tokenize(article_text)

print(f"Number of sentences: {len(sentences)}")

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")  # A lightweight, high-quality model

# Compute embeddings for all sentences
print("Generating sentence embeddings...")
embeddings = model.encode(sentences)

# Reduce embeddings to 2D with UMAP
print("Reducing embeddings to 2D with UMAP...")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="cosine")
embeddings_2d = reducer.fit_transform(embeddings)

# Cluster sentences using KMeans
print("Clustering sentences with KMeans...")
num_clusters = 5  # Change this to experiment with more or fewer clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Visualize clusters
plt.figure(figsize=(10, 8))
for cluster in range(num_clusters):
    points = embeddings_2d[labels == cluster]
    plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster}")
plt.title("Sentence Clusters from Article")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.show()

# Analyze cluster content
print("\n=== Cluster Content ===")
cluster_sentences = {i: [] for i in range(num_clusters)}
for sentence, label in zip(sentences, labels):
    cluster_sentences[label].append(sentence)

for cluster, sents in cluster_sentences.items():
    print(f"\nCluster {cluster} ({len(sents)} sentences):")
    for sent in sents[:5]:  # Show a few sample sentences from each cluster
        print(f"- {sent}")
