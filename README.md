# EXXA GSoC 2026 — Protoplanetary Disk Morphology Analysis

**Subproject:** EXXA2 — Exoplanet Atmosphere Characterization  
**Organization:** ML4Sci  

This project uses a Convolutional Autoencoder with unsupervised clustering (UMAP + HDBSCAN) to discover morphological patterns in synthetic ALMA observations of protoplanetary disks.

---

## Pipeline

```
FITS data → Preprocessing → Augmentation → ConvAutoencoder (train)
          → Latent Extraction → UMAP → HDBSCAN → Cluster Analysis
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/krishmannadal/UNSUPERVISED_ASTRONOMY.git
cd UNSUPERVISED_ASTRONOMY
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Place .fits data in data/continuum_data_subset/

# 3. Train the autoencoder
python -m training.train_autoencoder

# 4. Run the full pipeline (inference + clustering)
python EXXA_GSoC2026_Pipeline.py

# Or open the Jupyter Notebook version:
jupyter lab notebooks/EXXA_GSoC2026_Pipeline.ipynb
```

## Project Structure

```
├── dataset/
│   └── disk_dataset.py       # FITS loader with augmentation
├── models/
│   ├── autoencoder.py        # ConvAutoencoder (encode/decode API)
│   └── autoencoder_final.pth # Pretrained weights
├── training/
│   └── train_autoencoder.py  # GPU training with AMP + val split
├── embeddings/
│   └── latent_extract.py     # Latent vector extraction
├── clustering/
│   └── hdbscan_cluster.py    # UMAP + HDBSCAN clustering
├── analysis/
│   ├── radial_profile.py     # Radial brightness profiles
│   └── cluster_prototypes.py # Mean images per cluster
├── EXXA_GSoC2026_Pipeline.py # End-to-end pipeline script
├── notebooks/
│   └── EXXA_GSoC2026_Pipeline.ipynb  # Jupyter notebook version
└── generate_dummy_data.py    # Synthetic test data generator
```

## Key Features

| Feature | Implementation |
|:--------|:---------------|
| Data format | FITS data cubes, layer 0 extracted |
| Preprocessing | Percentile normalization [1%, 99%], resize to 256×256 |
| Augmentation | Random 90° rotation, H-flip, V-flip (prevents angle bias) |
| Model | 5-block CNN encoder/decoder with BatchNorm, 64-dim latent |
| Training | AMP mixed-precision, MSE + structural SSIM loss, train/val split |
| Metrics | MSE, Multiscale SSIM (pytorch-msssim) |
| Clustering | UMAP → HDBSCAN (density-based, no k required) |
| Validation | Silhouette score, radial profiles, cluster prototypes |

## Metrics (on synthetic data)

| Metric | Value |
|:-------|:------|
| MSE | 0.0127 |
| MS-SSIM | 0.4901 |
| Clusters | 4 |
| Silhouette | 0.92 |

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA optional)
- See `requirements.txt` for full list