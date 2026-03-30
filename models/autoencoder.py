"""
Convolutional Autoencoder for protoplanetary disk morphology learning.

Architecture:
    Encoder: 5 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool2d)
    Bottleneck: FC layers mapping to a compact latent vector
    Decoder: 5 transposed-convolutional blocks mirroring the encoder

The model accepts single-channel 256×256 images and compresses them
into a configurable latent space (default dim=64).
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with explicit encode/decode API
    so that latent representations are accessible without
    duplicating internal logic.
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # -------------------------------------------------
        # Encoder: 1×256×256 → 256×8×8
        # Added BatchNorm after every conv for training
        # stability and better gradient flow.
        # -------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        # Bottleneck
        self.fc_enc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        # -------------------------------------------------
        # Decoder: 256×8×8 → 1×256×256
        # Mirrors the encoder with transposed convolutions.
        # -------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    # =====================================================
    # Public API: encode / decode / forward
    # =====================================================

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image batch to its latent representation.

        Args:
            x: Tensor of shape (B, 1, 256, 256)

        Returns:
            z: Tensor of shape (B, latent_dim)
        """
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from latent vectors.

        Args:
            z: Tensor of shape (B, latent_dim)

        Returns:
            Reconstructed images of shape (B, 1, 256, 256)
        """
        x = self.fc_dec(z)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor):
        """Full forward pass: encode then decode.

        Returns:
            Tuple of (reconstructed, latent) so callers can
            access both the output image *and* the latent vector
            without calling encode() separately.
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z