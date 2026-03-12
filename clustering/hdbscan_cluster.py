import os
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import umap

# ---------------------------------
# Ensure outputs directory exists
# ---------------------------------
os.makedirs("outputs", exist_ok=True)


# ---------------------------------
# Load latent vectors
# ---------------------------------
latents = np.load("outputs/latents.npy")

print("Latent shape:", latents.shape)


# ---------------------------------
# HDBSCAN clustering
# ---------------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=4,
    metric="euclidean"
)

labels = clusterer.fit_predict(latents)

print("Cluster labels:")
print(labels)

# save cluster labels
np.save("outputs/cluster_labels.npy", labels)


# ---------------------------------
# UMAP dimensionality reduction
# ---------------------------------
reducer = umap.UMAP(
    n_neighbors=5,
    min_dist=0.3,
    random_state=42
)

embedding = reducer.fit_transform(latents)

x = embedding[:, 0]
y = embedding[:, 1]


# ---------------------------------
# Visualization
# ---------------------------------
plt.figure(figsize=(7,6))

plt.scatter(
    x,
    y,
    c=labels,
    cmap="tab10",
    s=80
)

plt.title("UMAP Projection of Disk Latent Space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.colorbar(label="Cluster label")

plt.tight_layout()

plt.savefig("outputs/umap_clusters.png")

plt.show()