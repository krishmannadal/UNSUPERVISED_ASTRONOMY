# EXXA GSoC 2026 — Protoplanetary Disk Morphology Analysis

**Subproject:** EXXA2 — Exoplanet Atmosphere Characterization  
**Organization:** ML4Sci  

This repository contains the full automated pipeline for the **EXXA2** GSoC tasks: **General Test** (Unsupervised Clustering) and **Image-Based Test** (Convolutional Autoencoder).

---

## 🛠 Step-by-Step Terminal Guide

Follow these steps to run the pipeline locally from your terminal.

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/krishmannadal/UNSUPERVISED_ASTRONOMY.git
cd UNSUPERVISED_ASTRONOMY

# Create and activate a fresh virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
The pipeline expects `.fits` files in the `data/continuum_data_subset/` directory.

**Option A - Real Data:**  
Place your official EXXA test FITS files (1250 microns ALMA continuum) into:  
`data/continuum_data_subset/`

**Option B - Synthetic Data (Quick Start):**  
If you do not have the official dataset yet, you can generate 100 synthetic disks to test the pipeline:
```bash
python generate_dummy_data.py
```

### 3. Model Training
Train the Convolutional Autoencoder. This script uses **GPU Mixed-Precision (AMP)** if a CUDA-enabled GPU is detected, and gracefully falls back to CPU otherwise.
```bash
python -m training.train_autoencoder
```

### 4. Running the Pipeline
Run the full inference, clustering, and analysis pipeline. This will:
1. Load the data and pre-trained model.
2. Calculate **MSE** and **MS-SSIM** reconstruction metrics.
3. Perform **UMAP** dimensionality reduction.
4. Execute **HDBSCAN** clustering.
5. Generate **Radial Brightness Profiles** and **Cluster Prototypes**.
```bash
python EXXA_GSoC2026_Pipeline.py
```

### 5. Viewing Results
After running, check the `outputs/` folder for:
- `umap_clusters.png`: Visualization of the discovered groups.
- `reconstruction_comparison.png`: Visual evaluation of the autoencoder.
- `radial_profiles_by_cluster.png`: Analysis of physical planet signatures.
- `cluster_0_prototypes.png` (etc.): Representative images for each cluster.

---

## 🧪 Google Colab (Preferred Method)
If you prefer to run this in the cloud (recommended by ML4Sci judges), use the standalone notebook:  
[Link to Standalone Colab Notebook](Your_Colab_Link_Here)

Or open the local version and upload it:  
`notebooks/colab_EXXA2.ipynb`

---

## 🏗 Project Architecture

| Component | Responsibility |
|:---|:---|
| `dataset/` | Robust FITS loading, 99th percentile normalization, automated augmentation |
| `models/` | ConvAutoencoder with BatchNorm and public `encode()`/`decode()` API |
| `training/` | GPU-optimized training with structural loss (MSE + SSIM) and Val split |
| `clustering/` | UMAP + HDBSCAN for density-based discovery |
| `analysis/` | Physical interpretability tools (Radial profiles, Prototypes) |

## 📊 Performance Summary (Synthetic Benchmark)
- **Clusters Detected:** 4 distinct morphological groups.
- **Silhouette Score:** 0.92 (High cluster separation).
- **Mean MS-SSIM:** 0.49.
- **Mean MSE:** 0.012.

---

## 📜 License & Acknowledgments
Developed for the GSoC 2026 EXXA program. Data simulation mimics standard ALMA configurations. Reference: Terry et al. (2022).