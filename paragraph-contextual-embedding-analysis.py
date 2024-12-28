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

# Split text into paragraphs and then sentences
paragraphs = [p.strip() for p in article_text.split("\n\n") if p.strip()]
sentence_to_paragraph = []  # To track which paragraph each sentence belongs to
sentences = []  # To store all sentences

# Map sentences to paragraphs
for paragraph_index, paragraph in enumerate(paragraphs):
    paragraph_sentences = nltk.sent_tokenize(paragraph)
    sentences.extend(paragraph_sentences)
    sentence_to_paragraph.extend([paragraph_index] * len(paragraph_sentences))

# Confirm mapping
print(f"Number of sentences: {len(sentences)}")
print(f"Number of paragraphs: {len(paragraphs)}")
print(f"Sentence-to-paragraph mapping: {sentence_to_paragraph[:10]}")  # Example

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

# Define the number of paragraphs per group
group_size = 3

# Create a grouped paragraph index
grouped_paragraph_index = [idx // group_size for idx in sentence_to_paragraph]

import matplotlib.colors as mcolors

# Define the number of groups based on the group size
num_groups = (len(paragraphs) + group_size - 1) // group_size  # Total groups

# Create a colormap with exactly num_groups colors
cmap = mcolors.ListedColormap(plt.cm.tab20.colors[:num_groups])

# Visualize grouped paragraphs across clusters
plt.figure(figsize=(7, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=grouped_paragraph_index, cmap=cmap, s=50)
plt.colorbar(scatter, label=f"Grouped Paragraph Index (Group Size: {group_size})")
plt.title("UMAP Visualization Colored by Grouped Paragraphs")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()


# Analyze paragraph-to-cluster mapping
paragraph_to_clusters = {i: [] for i in range(len(paragraphs))}

for sentence, cluster_label, paragraph_index in zip(sentences, labels, sentence_to_paragraph):
    paragraph_to_clusters[paragraph_index].append(cluster_label)

print("\n=== Paragraph-to-Cluster Mapping ===")
for paragraph_index, cluster_list in paragraph_to_clusters.items():
    unique_clusters = set(cluster_list)
    print(f"Paragraph {paragraph_index} is in clusters: {unique_clusters}")
    print(f"Cluster distribution: {cluster_list}")

# Analyze cluster content
print("\n=== Cluster Content ===")
cluster_sentences = {i: [] for i in range(num_clusters)}
for sentence, label in zip(sentences, labels):
    cluster_sentences[label].append(sentence)

for cluster, sents in cluster_sentences.items():
    print(f"\nCluster {cluster} ({len(sents)} sentences):")
    for sent in sents[:5]:  # Show a few sample sentences from each cluster
        print(f"- {sent}")
