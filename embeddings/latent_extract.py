"""
Latent vector extraction using a pretrained ConvAutoencoder.

Reads all FITS files from the dataset, encodes them, and saves
the resulting latent vectors to disk.

Usage:
    python -m embeddings.latent_extract
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.disk_dataset import DiskDataset
from models.autoencoder import ConvAutoencoder


def main():
    LATENT_DIM = 64
    DATA_DIR = "data/continuum_data_subset"
    MODEL_PATH = "models/autoencoder_final.pth"
    OUTPUT_DIR = "outputs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # No augmentation for inference — we want deterministic embeddings
    dataset = DiskDataset(DATA_DIR, image_size=256, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    latents = []
    filenames = []

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            z = model.encode(images)
            latents.append(z.cpu().numpy())

            # Track which file each latent came from
            for i in range(images.size(0)):
                global_idx = batch_idx * loader.batch_size + i
                filenames.append(dataset.get_filename(global_idx))

    latents = np.concatenate(latents, axis=0)

    np.save(os.path.join(OUTPUT_DIR, "latents.npy"), latents)
    np.save(os.path.join(OUTPUT_DIR, "filenames.npy"), np.array(filenames))

    print(f"Latent vectors saved: {latents.shape}")
    print(f"Filenames saved: {len(filenames)} entries")


if __name__ == "__main__":
    main()