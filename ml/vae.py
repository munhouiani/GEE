import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 69), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_idx):
        x = batch['features']
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=69,
                out_features=512
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=1024
            ),
            nn.ReLU(),
        )

        # output
        self.mu = nn.Linear(
            in_features=1024,
            out_features=100
        )
        self.logvar = nn.Linear(
            in_features=1024,
            out_features=100
        )

    def forward(self, x):
        h = self.fc(x)
        return self.mu(h), self.logvar(h)


class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=100,
                out_features=1024
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=1024,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # output
            nn.Linear(
                in_features=512,
                out_features=69
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)
