# GSoC 2026 Project Proposal
## EXXA: Exoplanet Atmosphere Characterization

**Name:** [Your Name]  
**Email:** [Your Email Address]  
**GitHub/GitLab Profile:** [Your Profile Link]  
**University/Institution:** [Your University]  
**Degree Program & Year:** [e.g., B.S. Computer Science, Junior]  
**Timezone:** [Your Timezone]  

---

## Abstract
Briefly summarize your proposed contribution. State that you are applying for the **EXXA2** subproject, focusing on protoplanetary disk characterization using deep learning and computer vision on synthetic and ALMA observational data. Provide a high-level overview of the Convolutional Autoencoder and clustering techniques (like HDBSCAN/UMAP) that you intend to develop/improve.

## 1. Project Detailed Description
Expand on the abstract. Mention the objective of uncovering how specific physical conditions manifest themselves in observational data (e.g., ALMA 1250 micron continuum observations). Discuss the use of Convolutional Autoencoders for dimensionality reduction and unsupervised learning. Explain how finding distinct morphological clusters correlates with the presence of planets (creating gaps/rings in the disks) rather than simple viewing angles, and detail how MS-SSIM and MSE will be used to ensure the autoencoder captures essential morphological features in the latent space.

### 1.1 Methodology
- **Data Preprocessing & Augmentation:** Detail how you will handle `.fits` data cubes (handling NaN values, 99th percentile normalization, resizing, rotation/flip augmentations).
- **Autoencoder Architecture:** Discuss your design choices for the Convolutional Autoencoder (encoder-decoder structure, latent vector dimension, activation layers, optimization method).
- **Latent Space Extraction & Evaluation:** Describe using PyTorch metrics (`pytorch-msssim` and MSE) to evaluate reconstruction accuracy.
- **Unsupervised Clustering:** Explain the pipeline for applying UMAP for dimensionality reduction followed by HDBSCAN to discover meaningful morphological clusters.

## 2. Deliverables
- A scalable, automated data pipeline for `.fits` data loading and preprocessing.
- A robust Convolutional Autoencoder implemented in PyTorch, capable of operating on withheld observational data.
- Clear clustering methodologies (UMAP+HDBSCAN) tied to actionable physical interpretations (e.g., detecting multi-planet systems).
- Extensive documentation, well-commented code, and a comprehensive Jupyter Notebook demonstrating the end-to-end pipeline.

## 3. Timeline / Milestones
*Note: Adjust this timeline based on whether you are doing the 12-week (standard) or 22-week (extended) schedule.*

- **Community Bonding Period (May 1 - May 25):** Engage with mentors, confirm dataset access, set up the development environment, and review relevant literature on protoplanetary disk formation.
- **Week 1-2 (May 26 - Jun 9):** Implement and refine the data pipeline. Integrate robust augmentations to handle orientation biases.
- **Week 3-4 (Jun 10 - Jun 24):** Finalize the Autoencoder architecture. Begin extended training runs and monitor MSE / MS-SSIM metrics tracking via MLflow.
- **Week 5-6 (Jun 25 - Jul 9):** Extract the latent representations. Tune UMAP and HDBSCAN parameters to ensure clusters map to physical characteristics rather than viewing angles.
- **Midterm Evaluation (Jul 10 - Jul 14):** Deliver the functional end-to-end pipeline script and present initial clustering results to mentors.
- **Week 7-9 (Jul 15 - Aug 4):** Refine clustering analysis and evaluate against the required metrics. Conduct rigorous testing on withheld and non-synthetic datasets if available.
- **Week 10-11 (Aug 5 - Aug 18):** Polish code, add comprehensive docstrings, build final Jupyter notebooks for tutorials and easy reproduction.
- **Week 12 (Aug 19 - Aug 26):** Final code cleanup, prepare the final project report, and submit all required evaluations.

## 4. Relevant Experience & Background
Provide a detailed background of your experience in Machine Learning (specifically PyTorch), Computer Vision, and any prior experience with Astronomical data (`astropy`, `.fits` handling). Link to any relevant projects, repositories, or prior open-source contributions. 
*Note: Mentioning the GSoC Test notebook that you completed will highly strengthen this section.*

## 5. Future Work
Discuss potential post-GSoC contributions, such as further integrating generative models (e.g., Diffusion models or VAEs) for data augmentation, or expanding the pipeline to other wavelengths.
