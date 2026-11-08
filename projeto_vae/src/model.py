"""VAE com MLP e função de custo (ELBO negativo)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.encoder = mlp([784] + hidden)
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)
        self.decoder = mlp([latent_dim] + hidden[::-1])
        self.out = nn.Linear(hidden[0], 784)  # logits dos pixels

    def encode(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.out(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparametrização
        return self.decode(z), mu, logvar


def vae_loss(logits, x, mu, logvar):
    """Soma de BCE (reconstrução) + KL no batch."""
    bce = F.binary_cross_entropy_with_logits(
        logits, x.view(x.size(0), -1), reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl
