import torch
import torch.nn as nn


class VAE(torch.nn.Module):
    """
    VAE model.

    z_size: (int) latent space dimension
    training: (bool)
    """

    def __init__(self, zdim=512, training=True):
        super(VAE, self).__init__()

        self.zdim = zdim
        self.training = training

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.mean_out = nn.Sequential(
            nn.Linear(3 * 8 * 256, self.zdim),
        )
        self.log_var_out = nn.Sequential(
            nn.Linear(3 * 8 * 256, self.zdim),
        )

        self.z_out = nn.Sequential(
            nn.Linear(self.zdim, 3 * 8 * 256),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        batch = x.shape[0]

        # encoder
        h = self.encoder(x)
        h = h.view(batch, -1)
        mean = self.mean_out(h)
        log_var = self.log_var_out(h)
        sigma = torch.exp(log_var / 2.0)
        eps = torch.randn_like(mean)

        if self.training:
            z = mean + sigma * eps

            # decoder
            h = self.z_out(z)
            h = h.view(batch, 3, 8, 256)
            h = self.decoder(h)

            return h, mean, log_var
        else:
            return mean

def create_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return optimizer

def loss_function(recon_batch, batch, mean, log_var, beta):
    batch_size = recon_batch.shape[0]
    BCE = nn.BCELoss(reduction='sum')(recon_batch, batch) / batch_size
    KLD = 0.5 * torch.sum(torch.exp(log_var) + mean ** 2 - 1 - log_var) / batch_size

    return BCE + beta * KLD