import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.disk_dataset import DiskDataset
from models.autoencoder import ConvAutoencoder


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Dataset
# -----------------------------
dataset = DiskDataset("data/continuum_data_subset", image_size=256)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)


# -----------------------------
# Load trained model
# -----------------------------
model = ConvAutoencoder(latent_dim=32).to(device)

model.load_state_dict(
    torch.load("models/autoencoder_final.pth", map_location=device)
)

model.eval()


# -----------------------------
# Extract latent vectors
# -----------------------------
latents = []

with torch.no_grad():

    for images in loader:

        images = images.to(device)

        x = model.encoder(images)
        x = model.flatten(x)

        z = model.fc_enc(x)

        latents.append(z.cpu().numpy())


latents = np.concatenate(latents, axis=0)


# -----------------------------
# Save latents
# -----------------------------
np.save("outputs/latents.npy", latents)

print("Latent vectors saved:", latents.shape)