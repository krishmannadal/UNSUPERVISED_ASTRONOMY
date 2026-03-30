# %% [markdown]
# # 🌌 EXXA GSoC 2026 — Standalone Colab Pipeline
# # Protoplanetary Disk Morphology & Unsupervised Clustering
#
# **Subproject:** EXXA2 — Exoplanet Atmosphere Characterization  
# **Author:** [Your Name]  
# **Date:** March 2026
#
# ---
#
# ## 🚀 1. Setup & Environment
# This cell checks for Google Colab and installs all necessary astronomical and ML dependencies.

# %%
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from astropy.io import fits

# Detect Colab
if 'google.colab' in sys.modules:
    print("Running on Google Colab. Installing dependencies...")
    !pip install -q astropy mlflow pytorch-msssim umap-learn hdbscan jupytext
else:
    print("Running in local environment.")

from pytorch_msssim import ms_ssim
from sklearn.metrics import silhouette_score
import hdbscan
import umap

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU Check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.backends.cudnn.benchmark = True
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Performance Tip: Enable 'GPU' in Edit -> Notebook settings.")

# %% [markdown]
# ---
# ## 📦 2. Data & Model Definitions (Standalone)
# All core logic is included here to ensure the notebook runs without external dependencies.

# %%
# @title Dataset Class & Model Architecture
class DiskDataset(Dataset):
    """FITS dataset loader with automated augmentation and Layer 0 extraction."""
    def __init__(self, data_dir, image_size=256, augment=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.fits')]
        if not self.files:
            print(f"⚠ Warning: No FITS files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def _normalize(self, data):
        data = np.nan_to_num(data)
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        if p99 > p1:
            data = (data - p1) / (p99 - p1)
        return data.astype(np.float32)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        with fits.open(path) as hdul:
            data = hdul[0].data
            # Handle 4D/3D cubes: (Layers, H, W)
            if data.ndim == 3: data = data[0]
            elif data.ndim == 4: data = data[0, 0]
        
        img = self._normalize(data)
        img_t = torch.from_numpy(img).unsqueeze(0) # [1, H, W]
        img_t = nn.functional.interpolate(img_t.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        
        if self.augment:
            if np.random.rand() > 0.5: img_t = torch.flip(img_t, [1]) # H-Flip
            if np.random.rand() > 0.5: img_t = torch.flip(img_t, [2]) # V-Flip
            k = np.random.randint(0, 4)
            img_t = torch.rot90(img_t, k, [1, 2])
            
        return img_t

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder with 64-dim Latent Space & BatchNorm."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_z = nn.Linear(256 * 8 * 8, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def encode(self, x): return self.fc_z(self.encoder(x))
    def decode(self, z): return self.decoder(self.decoder_input(z))
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# %% [markdown]
# ---
# ## 🧪 3. Synthetic Data Generator (Quick Start)
# Run this if you don't have the real ALMA data uploaded yet. It creates 100 realistic disks with gaps/rings.

# %%
def generate_fake_data(n=100, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        img = np.zeros((600, 600))
        y, x = np.ogrid[:600, :600]
        r = np.sqrt((x-300)**2 + (y-300)**2)
        # Disk + Gaps
        disk = np.exp(-r/150)
        n_planets = np.random.randint(0, 4)
        for _ in range(n_planets):
            gap_pos = np.random.randint(50, 200)
            disk *= (1 - 0.8 * np.exp(-(r-gap_pos)**2 / 100))
        img = disk + np.random.normal(0, 0.01, (600, 600))
        name = f"sim_disk_{i:03d}_p{n_planets}.fits"
        hdu = fits.PrimaryHDU(img.astype(np.float32))
        hdul = fits.HDUList([hdu])
        hdul.writeto(os.path.join(out_dir, name), overwrite=True)
    print(f"✅ Generated {n} synthetic disks in {out_dir}")

# Generate for demo
DATA_DIR = "colab_data"
if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
    generate_fake_data(100, DATA_DIR)

# %% [markdown]
# ---
# ## 🏋️ 4. Training
# Implements Structural Loss (MSE + SSIM) and Train/Val split.

# %%
# @title Run Training (Optional if pre-trained used)
LATENT_DIM = 64
EPOCHS = 20 # Shortened for demo
dataset = DiskDataset(DATA_DIR, augment=True)
train_size = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])

loader = DataLoader(train_set, batch_size=16, shuffle=True)
v_loader = DataLoader(val_set, batch_size=16)

model = ConvAutoencoder(latent_dim=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

print("Training started...")
for epoch in range(EPOCHS):
    model.train()
    for imgs in loader:
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        loss = crit(recon, imgs) + 0.1 * (1 - ms_ssim(recon, imgs, data_range=1.0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# %% [markdown]
# ---
# ## 🌌 5. Inference & Clustering
# Extract latents, project with UMAP, and cluster with HDBSCAN.

# %%
model.eval()
all_z, all_imgs = [], []
with torch.no_grad():
    for imgs in DataLoader(dataset, batch_size=16):
        imgs = imgs.to(device)
        z = model.encode(imgs)
        all_z.append(z.cpu().numpy())
        all_imgs.append(imgs.cpu().numpy())

latent_matrix = np.concatenate(all_z)
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
labels = clusterer.fit_predict(latent_matrix)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(latent_matrix)

# Visualize
plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=50, alpha=0.6)
plt.colorbar(label='Cluster ID')
plt.title(f'UMAP Projection of Disk Latent Space | Clusters: {len(np.unique(labels))}')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# %% [markdown]
# ---
# ## 📊 6. Performance Metrics
# Quantitative evaluation required by the Image-Based Test.

# %%
avg_mse = 0
avg_ssim = 0
with torch.no_grad():
    for imgs in DataLoader(dataset, batch_size=1):
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        avg_mse += nn.functional.mse_loss(recon, imgs).item()
        avg_ssim += ms_ssim(recon, imgs, data_range=1.0).item()

n = len(dataset)
print(f"Final Metrics across {n} images:")
print(f"Mean Squared Error (MSE): {avg_mse/n:.6f}")
print(f"Multiscale SSIM (MS-SSIM): {avg_ssim/n:.4f}")
