import torch
from torch import nn
from torch.nn import BCELoss

def reparameterize(mu, logvar):
    std = torch.exp(logvar / 2)
    z = torch.randn_like(logvar)
    return mu + z * std

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCELoss(reduction="sum")
        self.kl = lambda mu, logvar : torch.sum(-logvar + torch.exp(logvar) + mu ** 2 - 1) / 2
    
    def forward(self, x, recons, mu, logvar):
        L_recons = self.bce(recons, x)
        L_kl = self.kl(mu, logvar)
        return L_recons + L_kl

class VAEParameters(nn.Module):
    def __init__(self, embed_dim=256, latent_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(self.embed_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.embed_dim, self.latent_dim)
    
    def forward(self, x):
        return self.fc_mu(x), self.fc_logvar(x) 

class MNISTCVAE(nn.Module):
    def __init__(self, encoder, decoder, param):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.param = param

    def forward(self, x, c):
        embed = self.encoder(x, c)
        mu, logvar = self.param(embed)
        z = reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar