import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):

    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.fc_enc = nn.Linear(256*8*8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*8*8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,2,stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.flatten(x)

        z = self.fc_enc(x)

        x = self.fc_dec(z)
        x = x.view(-1,256,8,8)

        x = self.decoder(x)

        return x