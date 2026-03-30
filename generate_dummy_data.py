"""
Generate synthetic protoplanetary disk FITS files for testing.

Creates data cubes in the same format as the EXXA test data:
4 layers of 600×600, with layer 0 containing the continuum
observation. Various disk morphologies are generated:
    - Smooth disks (no planets)
    - Single-gap disks (1 planet)
    - Multi-gap disks (2–3 planets)
    - Inclined disks (varying viewing angles)

Usage:
    python generate_dummy_data.py
"""

import os
import numpy as np
from astropy.io import fits


def make_disk_image(size=600, n_planets=0, inclination=0.0, noise_level=0.05):
    """Create a single synthetic protoplanetary disk image.

    Args:
        size:        Image side length in pixels.
        n_planets:   Number of planets (0–3). Each creates a gap.
        inclination: Viewing angle in degrees (0 = face-on).
        noise_level: Gaussian noise standard deviation.

    Returns:
        2D numpy array (float32) representing the disk observation.
    """
    y, x = np.ogrid[-size // 2: size // 2, -size // 2: size // 2]
    y = y.astype(np.float64)
    x = x.astype(np.float64)

    # Apply inclination (stretch along y-axis)
    cos_inc = np.cos(np.radians(inclination))
    y_eff = y / max(cos_inc, 0.3)  # Prevent division collapse
    r = np.sqrt(x ** 2 + y_eff ** 2)

    # Central star (Gaussian core)
    star = np.exp(-r ** 2 / 12 ** 2) * 8.0

    # Extended disk emission (power-law-ish profile with taper)
    disk = 2.0 * np.exp(-r / (size * 0.2)) * (1.0 / (1.0 + (r / 10) ** 2))

    image = star + disk

    # Planetary gaps — radial dips at specific locations
    gap_radii = np.linspace(40, size * 0.3, n_planets + 2)[1:-1]  # Spread evenly
    for gap_r in gap_radii[:n_planets]:
        gap_width = np.random.uniform(5, 15)
        gap_depth = np.random.uniform(0.3, 0.8)
        gap_profile = 1.0 - gap_depth * np.exp(-((r - gap_r) ** 2) / (2 * gap_width ** 2))
        image *= gap_profile

    # Spiral arms (occasionally, for multi-planet systems)
    if n_planets >= 2 and np.random.rand() > 0.5:
        theta = np.arctan2(y_eff, x)
        spiral = 0.1 * np.sin(2 * theta + r / 20)
        image += spiral * np.exp(-r / (size * 0.15))

    # Observational noise
    image += np.random.normal(0, noise_level, image.shape)
    image = np.maximum(image, 0).astype(np.float32)

    return image


def create_synthetic_data(output_dir="data/continuum_data_subset", num_samples=30):
    """Generate a diverse set of synthetic disk observations.

    Creates FITS files with 4-layer data cubes to match the
    real EXXA test data format. Layer 0 is the continuum image.
    """
    os.makedirs(output_dir, exist_ok=True)

    configs = []
    for i in range(num_samples):
        # Vary the number of planets and viewing angle
        n_planets = np.random.choice([0, 0, 1, 1, 1, 2, 2, 3])
        inclination = np.random.uniform(0, 60)
        configs.append((n_planets, inclination))

    for i, (n_planets, inc) in enumerate(configs):
        # Layer 0: the continuum observation (only relevant one)
        layer0 = make_disk_image(
            size=600, n_planets=n_planets,
            inclination=inc, noise_level=0.05,
        )

        # Layers 1–3: dummy data (not used, but present in real cubes)
        layers = [layer0]
        for _ in range(3):
            layers.append(np.zeros_like(layer0))

        data_cube = np.stack(layers, axis=0)  # Shape: (4, 600, 600)

        # Write FITS
        header = fits.Header()
        header["OBJECT"] = f"synthetic_disk_{i}"
        header["NPLANETS"] = n_planets
        header["INCLIN"] = round(inc, 1)
        header.add_comment("Synthetic protoplanetary disk for EXXA GSoC 2026 testing")

        hdu = fits.PrimaryHDU(data_cube, header=header)
        filepath = os.path.join(output_dir, f"disk_{i:04d}.fits")
        hdu.writeto(filepath, overwrite=True)

    print(f"Created {num_samples} synthetic FITS files in {output_dir}")
    print(f"Planet distribution: {[c[0] for c in configs]}")


if __name__ == "__main__":
    create_synthetic_data()
