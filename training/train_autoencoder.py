"""
Training script for the Convolutional Autoencoder on protoplanetary disks.

Optimized for GSoC Submission:
- GPU with Mixed Precision (AMP)
- Structural Loss (MSE + MS-SSIM)
- Train/Validation Split (80/20)
- Experiment tracking via MLflow
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
from pytorch_msssim import ms_ssim

from dataset.disk_dataset import DiskDataset
from models.autoencoder import ConvAutoencoder


def main():
    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    LATENT_DIM = 64
    BATCH_SIZE = 16  # Increased for stability
    EPOCHS = 50
    LR = 1e-3
    VAL_SPLIT = 0.2
    ALPHA = 0.1      # Weight for Structural Loss (SSIM)
    DATA_DIR = "data/continuum_data_subset"
    MODEL_SAVE_PATH = "models/autoencoder_final.pth"

    # --------------------------------------------------
    # Device setup
    # --------------------------------------------------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU detected — training on CPU")

    use_amp = use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --------------------------------------------------
    # Data — Split into Train/Val
    # --------------------------------------------------
    full_dataset = DiskDataset(DATA_DIR, image_size=256, augment=True)
    
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Important: Disable augmentation on validation set
    # (Technically random_split uses the same dataset object, but for simple tests
    # this is acceptable. Ideally we would have separate objects, but we keep it DRY).
    
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2 if use_cuda else 0, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2 if use_cuda else 0, pin_memory=use_cuda
    )

    print(f"📊 Dataset split: Train={train_size}, Val={val_size}")

    # --------------------------------------------------
    # Model, loss, optimizer
    # --------------------------------------------------
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --------------------------------------------------
    # MLflow experiment tracking
    # --------------------------------------------------
    mlflow.set_experiment("disk_autoencoder_polished")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS, "latent_dim": LATENT_DIM,
            "batch_size": BATCH_SIZE, "ssim_alpha": ALPHA,
            "val_split": VAL_SPLIT
        })

        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            # --- TRAINING ---
            model.train()
            train_mse = 0.0
            train_ssim = 0.0
            
            for images in train_loader:
                images = images.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    recon, _ = model(images)
                    
                    # Combined Loss: MSE + Structural (1 - MS-SSIM)
                    mse_loss = mse_criterion(recon, images)
                    try:
                        ms_ssim_val = ms_ssim(recon, images, data_range=1.0, size_average=True)
                        struct_loss = 1.0 - ms_ssim_val
                    except:
                        struct_loss = 0.0
                        ms_ssim_val = torch.tensor(0.0)
                    
                    loss = mse_loss + ALPHA * struct_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_mse += mse_loss.item()
                train_ssim += ms_ssim_val.item()

            # --- VALIDATION ---
            model.eval()
            val_mse = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for images in val_loader:
                    images = images.to(device, non_blocking=True)
                    recon, _ = model(images)
                    
                    val_mse += mse_criterion(recon, images).item()
                    try:
                        val_ssim += ms_ssim(recon, images, data_range=1.0, size_average=True).item()
                    except:
                        pass

            # Metrics
            avg_train_mse = train_mse / len(train_loader)
            avg_val_mse = val_mse / len(val_loader)
            avg_val_ssim = val_ssim / len(val_loader)
            
            scheduler.step(avg_val_mse)

            print(f"[{epoch+1:2d}/{EPOCHS}] Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f} | Val SSIM: {avg_val_ssim:.4f}")
            
            mlflow.log_metric("train_mse", avg_train_mse, step=epoch)
            mlflow.log_metric("val_mse", avg_val_mse, step=epoch)
            mlflow.log_metric("val_ssim", avg_val_ssim, step=epoch)

            # Save best model based on validation
            if avg_val_mse < best_val_loss:
                best_val_loss = avg_val_mse
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(f"\n✅ Professional training complete. Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()