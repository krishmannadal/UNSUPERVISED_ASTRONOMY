import numpy as np
import matplotlib.pyplot as plt
from dataset.disk_dataset import DiskDataset

# ---------------------------------
# Load dataset
# ---------------------------------
dataset = DiskDataset("data/continuum_data_subset", image_size=256)

# ---------------------------------
# Load cluster labels
# ---------------------------------
labels = np.load("outputs/cluster_labels.npy")

clusters = np.unique(labels)

print("Clusters found:", clusters)


# ---------------------------------
# Compute prototype image per cluster
# ---------------------------------
for c in clusters:

    if c == -1:
        continue   # skip noise cluster

    indices = np.where(labels == c)[0]

    images = []

    for idx in indices:
        img = dataset[idx].numpy()[0]  # remove channel dimension
        images.append(img)

    images = np.stack(images)

    prototype = np.mean(images, axis=0)

    # ---------------------------------
    # Plot prototype disk
    # ---------------------------------
    plt.figure(figsize=(4,4))

    plt.imshow(prototype, cmap="inferno")

    plt.title(f"Cluster {c} Prototype ({len(indices)} disks)")
    plt.axis("off")

    plt.show()