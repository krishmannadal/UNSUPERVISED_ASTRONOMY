"""
HDBSCAN clustering of autoencoder latent vectors with UMAP visualization.

Reads latent vectors from outputs/latents.npy, performs UMAP
dimensionality reduction, clusters with HDBSCAN, and saves both
the cluster labels and a publication-quality scatter plot.

Usage:
    python -m clustering.hdbscan_cluster
"""

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

# Load filenames if available (for labeling)
filenames_path = "outputs/filenames.npy"
filenames = np.load(filenames_path, allow_pickle=True) if os.path.exists(filenames_path) else None


# ---------------------------------
# UMAP dimensionality reduction
# ---------------------------------
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42,
)

embedding = reducer.fit_transform(latents)


# ---------------------------------
# HDBSCAN clustering
# ---------------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=4,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom",
)

labels = clusterer.fit_predict(embedding)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"Clusters found: {n_clusters}")
print(f"Noise points:   {n_noise}")

# Save cluster labels
np.save("outputs/cluster_labels.npy", labels)


# ---------------------------------
# Cluster summary table
# ---------------------------------
print("\n┌──────────┬──────────┐")
print("│ Cluster  │  Count   │")
print("├──────────┼──────────┤")
for c_id in sorted(set(labels)):
    label_str = "noise" if c_id == -1 else str(c_id)
    count = (labels == c_id).sum()
    print(f"│ {label_str:>8s} │ {count:>8d} │")
print("└──────────┴──────────┘")


# ---------------------------------
# Visualization
# ---------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

# Plot noise points in gray
noise_mask = labels == -1
if noise_mask.any():
    ax.scatter(
        embedding[noise_mask, 0],
        embedding[noise_mask, 1],
        c="lightgray",
        marker="x",
        s=40,
        alpha=0.5,
        label="Noise",
    )

# Plot clusters with distinct colors
cluster_mask = ~noise_mask
scatter = ax.scatter(
    embedding[cluster_mask, 0],
    embedding[cluster_mask, 1],
    c=labels[cluster_mask],
    cmap="Set1",
    s=90,
    edgecolors="black",
    linewidths=0.5,
)

ax.set_title("UMAP Projection of Disk Latent Space", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.legend(fontsize=10)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label("Cluster ID", fontsize=11)

plt.tight_layout()
plt.savefig("outputs/umap_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: outputs/umap_clusters.png")
print("Saved: outputs/cluster_labels.npy")