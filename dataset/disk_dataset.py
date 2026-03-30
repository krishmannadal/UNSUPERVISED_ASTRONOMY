"""
Dataset class for loading protoplanetary disk FITS observations.

Handles:
    - 4-layer ALMA data cubes (explicitly selects layer 0)
    - Percentile-based normalization (robust to outliers)
    - Resizing to a fixed spatial resolution
    - Automated augmentation (random rotations, flips) to prevent
      the model from clustering by viewing angle instead of
      physical disk morphology.
"""

import os
import glob
import numpy as np
from astropy.io import fits
import torch
from torch.utils.data import Dataset
import cv2


class DiskDataset(Dataset):
    """PyTorch Dataset for .fits protoplanetary disk images.

    Args:
        folder_path: Directory containing .fits files.
        image_size:  Target spatial resolution (square).
        augment:     Whether to apply random rotations/flips.
                     Should be True for training, False for
                     inference/evaluation.
    """

    def __init__(self, folder_path, image_size=256, augment=False):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.fits")))
        self.image_size = image_size
        self.augment = augment

        if len(self.files) == 0:
            raise ValueError(
                f"No FITS files found in '{folder_path}'. "
                "Check that the path contains *.fits files."
            )

        print(f"Loaded {len(self.files)} FITS files from {folder_path}")

    def __len__(self):
        return len(self.files)

    # -------------------------
    # Load FITS
    # -------------------------

    def _load_fits(self, path):
        """Load a FITS file and extract the first spatial plane.

        Per the EXXA specification, each image is a data cube
        containing 4 (600×600) layers.  Only layer 0 is relevant.
        We handle both multi-dimensional cubes and plain 2D images.
        """
        with fits.open(path) as hdul:
            data = hdul[0].data

        if data is None:
            raise ValueError(f"No data in primary HDU of {path}")

        data = data.astype(np.float32)

        # Replace NaNs/Infs early (common in ALMA data)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Explicitly select layer 0 for each extra dimension
        # This correctly handles shapes like (4, 600, 600) or
        # (1, 4, 600, 600) instead of blindly iterating data[0].
        while data.ndim > 2:
            data = data[0]

        if data.ndim != 2:
            raise ValueError(
                f"Could not reduce FITS to 2D from shape {data.shape}: {path}"
            )

        return data

    # -------------------------
    # Normalize
    # -------------------------

    def _normalize(self, image):
        """Percentile-based normalization to [0, 1].

        Uses the 1st/99th percentiles to clip extreme outliers
        before rescaling.  This is critical for ALMA data
        which can have very bright central pixels.
        """
        image = np.nan_to_num(image, nan=0.0)

        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)

        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1 + 1e-8)

        return image

    # -------------------------
    # Resize
    # -------------------------

    def _resize(self, image):
        """Resize to the target square resolution."""
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        return image

    # -------------------------
    # Augmentation
    # -------------------------

    def _augment_image(self, image):
        """Apply random geometric augmentations.

        These are orientation-invariant transforms designed to
        prevent the encoder from learning viewing angle as a
        primary feature (a known pitfall noted by the EXXA
        evaluators: 'Beware of simply clustering the disks
        by viewing angle').

        Transforms applied (each with 50% probability):
            1. Random 90° rotation (0/90/180/270°)
            2. Horizontal flip
            3. Vertical flip
        """
        # Random k×90° rotation
        k = np.random.randint(0, 4)
        image = np.rot90(image, k=k).copy()

        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()

        return image

    # -------------------------
    # Main fetch
    # -------------------------

    def __getitem__(self, index):
        path = self.files[index]

        image = self._load_fits(path)
        image = self._normalize(image)
        image = self._resize(image)

        if self.augment:
            image = self._augment_image(image)

        # Add channel dimension → (1, H, W)
        image = np.expand_dims(image, axis=0)
        tensor = torch.from_numpy(image).float()

        return tensor

    def get_filename(self, index):
        """Return the basename of the FITS file at the given index."""
        return os.path.basename(self.files[index])
