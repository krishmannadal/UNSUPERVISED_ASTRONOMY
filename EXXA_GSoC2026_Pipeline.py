# %% [markdown]
# # EXXA GSoC 2026 — General Test & Image-Based Test
# # Unsupervised Clustering + Autoencoder Pipeline
#
# **Subproject:** EXXA2 — Exoplanet Atmosphere Characterization  
# **Author:** [Your Name]  
# **Date:** March 2026
#
# ---
#
# ## Overview
#
# This notebook implements an end-to-end pipeline for the two EXXA GSoC tests:
#
# **General Test:** Unsupervised clustering of synthetic ALMA continuum
# observations (1250 µm) of protoplanetary disks. The goal is to discover
# morphological groups that correspond to physical properties — in particular,
# the number of planets and their signatures (gaps, rings, spirals).
#
# **Image-Based Test:** A Convolutional Autoencoder that reconstructs the
# input disk images with an accessible latent space. Evaluated quantitatively
# with MSE and Multiscale SSIM (MS-SSIM).
#
# ### Pipeline Summary
# ```
# FITS data → preprocessing → augmentation → ConvAutoencoder (training)
#           → latent extraction → UMAP → HDBSCAN → cluster analysis
# ```
#
# ### Key Design Decisions
# - **Automated augmentation** (random 90° rotations + flips) to prevent
#   the model from trivially clustering disks by viewing angle.
# - **BatchNorm** in the autoencoder for stable training with varying
#   flux distributions.
# - **Percentile normalization** (1st–99th) to handle ALMA's extreme
#   dynamic range.

# %% [markdown]
# ---
# ## 1. Environment Setup

# %%
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
from sklearn.metrics import silhouette_score
import hdbscan
import umap

warnings.filterwarnings("ignore", category=FutureWarning)

# Allow imports from project root
sys.path.insert(0, os.path.abspath('.'))

from dataset.disk_dataset import DiskDataset
from models.autoencoder import ConvAutoencoder
from analysis.radial_profile import radial_profile

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("Device: CPU")

# ──────────────────────────────────────────────
# Configuration — change these paths as needed
# ──────────────────────────────────────────────
DATA_DIR    = "data/continuum_data_subset"
MODEL_PATH  = "models/autoencoder_final.pth"
OUTPUT_DIR  = "outputs"
LATENT_DIM  = 64
BATCH_SIZE  = 8
IMAGE_SIZE  = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ---
# ## 2. Data Loading & Inspection
#
# We load all `.fits` files from the dataset directory.
# Each file is a data cube; we extract **layer 0** (the continuum
# observation) as specified by the EXXA test description.

# %%
# No augmentation for inference — we need deterministic results
dataset = DiskDataset(DATA_DIR, image_size=IMAGE_SIZE, augment=False)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=use_cuda,   # faster CPU→GPU transfer
    num_workers=2 if use_cuda else 0,
)

print(f"Total samples: {len(dataset)}")
print(f"Batches:       {len(dataloader)}")

# %%
# Visualize a sample of raw disk images
n_show = min(8, len(dataset))
fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 3))
if n_show == 1:
    axes = [axes]

for i in range(n_show):
    img = dataset[i][0].numpy()
    axes[i].imshow(img, cmap="inferno", origin="lower")
    axes[i].set_title(dataset.get_filename(i)[:12], fontsize=8)
    axes[i].axis("off")

fig.suptitle("Sample Disk Observations (layer 0, normalized)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_inputs.png"), dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 3. Model Architecture
#
# The `ConvAutoencoder` has:
# - **Encoder:** 5 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool2d),
#   reducing 256×256 → 8×8 spatially, then a fully-connected layer
#   projects to the latent vector of dimension 64.
# - **Decoder:** Mirrors the encoder with transposed convolutions.
# - **Latent access:** `model.encode(x)` returns the latent vector `z`,
#   and `model.decode(z)` reconstructs the image.

# %%
model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        print("✓ Loaded pretrained model from", MODEL_PATH)
    except RuntimeError as e:
        print(f"⚠ Could not load pretrained weights (architecture mismatch).")
        print(f"  Reason: {e}")
        print("  Continuing with randomly initialized weights for demo.")
        print("  Re-train with: python -m training.train_autoencoder")
else:
    print("⚠ Pretrained model not found — using random weights for demo.")
    print("  Train with: python -m training.train_autoencoder")

model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

# %% [markdown]
# ---
# ## 4. Inference: Reconstruction + Latent Extraction
#
# We pass every image through the autoencoder and collect:
# 1. **Latent vectors** for clustering.
# 2. **Reconstructed images** for metric computation.

# %%
all_latents = []
all_originals = []
all_reconstructed = []

mse_per_sample = []
msssim_per_sample = []

mse_fn = nn.MSELoss(reduction="none")

with torch.no_grad():
    for imgs in dataloader:
        imgs = imgs.to(device, non_blocking=True)

        z = model.encode(imgs)
        recon = model.decode(z)

        all_latents.append(z.cpu().numpy())
        all_originals.append(imgs.cpu())
        all_reconstructed.append(recon.cpu())

        # Per-sample MSE
        sample_mse = mse_fn(recon, imgs).mean(dim=[1, 2, 3])
        mse_per_sample.append(sample_mse.cpu().numpy())

        # Per-sample MS-SSIM
        try:
            sample_ssim = ms_ssim(
                recon, imgs, data_range=1.0, size_average=False
            )
            msssim_per_sample.append(sample_ssim.cpu().numpy())
        except RuntimeError as e:
            # MS-SSIM requires min spatial size of 160px for default scales
            print(f"MS-SSIM skipped for batch (reason: {e})")
            msssim_per_sample.append(np.zeros(imgs.size(0)))

# Stack everything
latents_arr = np.vstack(all_latents)
originals   = torch.cat(all_originals, dim=0)
recons      = torch.cat(all_reconstructed, dim=0)
mse_arr     = np.concatenate(mse_per_sample)
msssim_arr  = np.concatenate(msssim_per_sample)

print(f"Latent matrix shape: {latents_arr.shape}")

# %% [markdown]
# ---
# ## 5. Quantitative Metrics (Image-Based Test)
#
# The EXXA Image-Based Test requires:
# - **MSE** between input and output
# - **Multiscale SSIM** between input and output

# %%
print("╔════════════════════════════════════════════════╗")
print("║        RECONSTRUCTION METRICS SUMMARY          ║")
print("╠════════════════════════════════════════════════╣")
print(f"║  Mean MSE:     {mse_arr.mean():.6f} ± {mse_arr.std():.6f}    ║")
print(f"║  Mean MS-SSIM: {msssim_arr.mean():.6f} ± {msssim_arr.std():.6f}    ║")
print(f"║  Best MSE:     {mse_arr.min():.6f}                   ║")
print(f"║  Best MS-SSIM: {msssim_arr.max():.6f}                   ║")
print("╚════════════════════════════════════════════════╝")

# Per-sample metric table
print("\n  Sample-level metrics:")
for i in range(len(mse_arr)):
    fname = dataset.get_filename(i)
    print(f"    {fname:30s}  MSE={mse_arr[i]:.6f}  MS-SSIM={msssim_arr[i]:.4f}")

# %% [markdown]
# ---
# ## 6. Reconstruction Visualization
#
# Side-by-side comparison of original and reconstructed disks.

# %%
n_vis = min(6, len(dataset))
fig, axes = plt.subplots(3, n_vis, figsize=(3 * n_vis, 9))

for i in range(n_vis):
    orig = originals[i, 0].numpy()
    rec  = recons[i, 0].numpy()
    diff = np.abs(orig - rec)

    axes[0, i].imshow(orig, cmap="inferno", origin="lower")
    axes[0, i].set_title(f"Input {i}", fontsize=9)
    axes[0, i].axis("off")

    axes[1, i].imshow(rec, cmap="inferno", origin="lower")
    axes[1, i].set_title(f"Recon (MSE={mse_arr[i]:.4f})", fontsize=9)
    axes[1, i].axis("off")

    axes[2, i].imshow(diff, cmap="hot", origin="lower")
    axes[2, i].set_title("| Residual |", fontsize=9)
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=11)
axes[1, 0].set_ylabel("Reconstruction", fontsize=11)
axes[2, 0].set_ylabel("Residual", fontsize=11)

fig.suptitle("Autoencoder Reconstruction Quality", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reconstruction_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Latent Space Access Demonstration
#
# The Image-Based Test requires that *"a user should be able to feed in
# an image and access it when it has been encoded in the latent space."*
#
# Here we demonstrate this API:

# %%
# Pick a single sample
sample_idx = 0
single_image = dataset[sample_idx].unsqueeze(0).to(device)
print(f"Input shape:  {single_image.shape}")

with torch.no_grad():
    latent_vector = model.encode(single_image)
    print(f"Latent shape: {latent_vector.shape}")
    print(f"Latent values (first 10): {latent_vector[0, :10].cpu().numpy()}")

    # Decode back
    decoded = model.decode(latent_vector)
    print(f"Output shape: {decoded.shape}")

# %% [markdown]
# ---
# ## 8. Dimensionality Reduction (UMAP)

# %%
print("Running UMAP projection...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42,
)
embedding_2d = reducer.fit_transform(latents_arr)
print(f"UMAP embedding shape: {embedding_2d.shape}")

# %% [markdown]
# ---
# ## 9. Unsupervised Clustering (HDBSCAN)
#
# HDBSCAN identifies clusters of varying density without requiring
# a pre-specified number of clusters — ideal when the number of
# distinct disk morphologies is unknown.

# %%
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(embedding_2d)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"Clusters discovered:  {n_clusters}")
print(f"Noise points:         {n_noise}")
print(f"Cluster membership probabilities (mean): {clusterer.probabilities_.mean():.3f}")

# Save results
np.save(os.path.join(OUTPUT_DIR, "latents.npy"), latents_arr)
np.save(os.path.join(OUTPUT_DIR, "cluster_labels.npy"), labels)
np.save(os.path.join(OUTPUT_DIR, "umap_embedding.npy"), embedding_2d)
print("Saved: latents.npy, cluster_labels.npy, umap_embedding.npy")

# %% [markdown]
# ---
# ## 10. Cluster Analysis & Visualization
#
# The evaluators have stated: *"Models will be judged on the clarity of
# clusters produced and the properties that the clusters find."*
#
# We provide multiple views: UMAP scatter, per-cluster prototypes,
# per-cluster radial profiles, and a summary statistics table.

# %%
# ── 10a. UMAP scatter plot with cluster coloring ──
fig, ax = plt.subplots(figsize=(10, 8))

noise_mask = labels == -1
if noise_mask.any():
    ax.scatter(
        embedding_2d[noise_mask, 0], embedding_2d[noise_mask, 1],
        c="lightgray", marker="x", s=50, alpha=0.6, label="Noise", zorder=1,
    )

cluster_ids = sorted(set(labels) - {-1})
colors = plt.cm.Set1(np.linspace(0, 1, max(len(cluster_ids), 1)))

for ci, color in zip(cluster_ids, colors):
    mask = labels == ci
    ax.scatter(
        embedding_2d[mask, 0], embedding_2d[mask, 1],
        color=color, s=100, edgecolors="k", linewidths=0.5,
        label=f"Cluster {ci} (n={mask.sum()})", zorder=2,
    )
    # Annotate filenames
    for j in np.where(mask)[0]:
        ax.annotate(
            dataset.get_filename(j)[:8], (embedding_2d[j, 0], embedding_2d[j, 1]),
            fontsize=6, alpha=0.7, ha="center", va="bottom",
        )

ax.set_title("UMAP Projection — Protoplanetary Disk Morphology Clusters",
             fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.legend(loc="best", fontsize=9, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_clusters.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── 10b. Silhouette score (cluster quality metric) ──
if n_clusters >= 2:
    non_noise = labels != -1
    sil = silhouette_score(embedding_2d[non_noise], labels[non_noise])
    print(f"Silhouette Score: {sil:.4f}  (1.0=perfect, 0.0=overlapping)")
else:
    print("Not enough clusters for silhouette score.")

# %%
# ── 10c. Cluster summary table ──
print("\n┌──────────┬───────┬────────────────┬─────────────────┐")
print("│ Cluster  │ Count │ Mean MSE       │ Mean MS-SSIM    │")
print("├──────────┼───────┼────────────────┼─────────────────┤")
for c_id in sorted(set(labels)):
    mask = labels == c_id
    label_str = "noise" if c_id == -1 else str(c_id)
    count = mask.sum()
    c_mse = mse_arr[mask].mean() if mask.any() else 0.0
    c_ssim = msssim_arr[mask].mean() if mask.any() else 0.0
    print(f"│ {label_str:>8s} │ {count:>5d} │ {c_mse:>14.6f} │ {c_ssim:>15.6f} │")
print("└──────────┴───────┴────────────────┴─────────────────┘")

# %%
# ── 10d. Prototype images per cluster ──
# Shows the mean image for each cluster, giving an immediate
# visual sense of what morphology the cluster represents.

for c_id in cluster_ids:
    indices = np.where(labels == c_id)[0]
    cluster_images = []
    for idx in indices:
        cluster_images.append(originals[idx, 0].numpy())

    prototype = np.mean(cluster_images, axis=0)

    n_members = min(4, len(indices))
    fig, axes = plt.subplots(1, n_members + 1, figsize=(3.5 * (n_members + 1), 3.5))

    axes[0].imshow(prototype, cmap="inferno", origin="lower")
    axes[0].set_title(f"Cluster {c_id}\nPrototype (mean)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for j in range(n_members):
        axes[j + 1].imshow(originals[indices[j], 0].numpy(), cmap="inferno", origin="lower")
        axes[j + 1].set_title(dataset.get_filename(indices[j])[:15], fontsize=8)
        axes[j + 1].axis("off")

    plt.suptitle(f"Cluster {c_id} — {len(indices)} members", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_{c_id}_prototypes.png"), dpi=120, bbox_inches="tight")
    plt.show()

# %%
# ── 10e. Per-cluster radial brightness profiles ──
# Radial profiles are a standard analysis tool in disk research.
# Gaps caused by planets appear as dips in the radial profile.

fig, ax = plt.subplots(figsize=(10, 6))

for c_id in cluster_ids:
    indices = np.where(labels == c_id)[0]
    profiles = []
    for idx in indices:
        rp = radial_profile(originals[idx, 0].numpy())
        profiles.append(rp)

    # Pad profiles to same length and average
    max_len = max(len(p) for p in profiles)
    padded = [np.pad(p, (0, max_len - len(p)), constant_values=0) for p in profiles]
    mean_profile = np.mean(padded, axis=0)

    ax.plot(mean_profile, label=f"Cluster {c_id} (n={len(indices)})", linewidth=2)

ax.set_title("Mean Radial Brightness Profiles by Cluster", fontsize=14, fontweight="bold")
ax.set_xlabel("Radius (pixels from center)", fontsize=12)
ax.set_ylabel("Mean Normalized Brightness", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "radial_profiles_by_cluster.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 11. Summary
#
# | Component             | Implementation                              |
# |:----------------------|:--------------------------------------------|
# | Data format           | FITS data cubes, layer 0 extracted           |
# | Preprocessing         | Percentile norm [1%, 99%], resize to 256×256 |
# | Augmentation          | Random 90° rotation, H-flip, V-flip         |
# | Encoder               | 5-block CNN + BatchNorm → 64-dim latent      |
# | Decoder               | 5-block transpose CNN + Sigmoid              |
# | Reconstruction metric | MSE, MS-SSIM (pytorch-msssim)                |
# | Dim. reduction        | UMAP (2D)                                    |
# | Clustering            | HDBSCAN (density-based, no k required)       |
# | Cluster validation    | Silhouette score, radial profiles, prototypes|
#
# The clusters discovered correspond to morphological
# differences in the disk structure. Radial profile analysis
# reveals whether each cluster exhibits gap/ring signatures
# consistent with the presence of one or more planets.

# %%
print("═" * 50)
print("  Pipeline complete.")
print(f"  Clusters found:  {n_clusters}")
print(f"  Mean MSE:        {mse_arr.mean():.6f}")
print(f"  Mean MS-SSIM:    {msssim_arr.mean():.6f}")
print("═" * 50)
