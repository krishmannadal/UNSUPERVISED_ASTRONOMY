import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from dataset.disk_dataset import DiskDataset
from models.autoencoder import ConvAutoencoder


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DiskDataset("data/continuum_data_subset")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ConvAutoencoder(latent_dim=32).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    mlflow.set_experiment("disk_autoencoder")

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("latent_dim", 32)

        for epoch in range(epochs):

            total_loss = 0

            for images in loader:

                images = images.to(device)

                outputs = model(images)

                loss = criterion(outputs, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            print(f"Epoch {epoch} Loss {avg_loss}")

            mlflow.log_metric("loss", avg_loss, step=epoch)

        mlflow.pytorch.log_model(model, "autoencoder")
        torch.save(model.state_dict(), "models/autoencoder_final.pth")

        print("Model saved to models/autoencoder_final.pth")


if __name__ == "__main__":
    main()