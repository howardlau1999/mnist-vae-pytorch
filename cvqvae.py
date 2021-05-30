from torch import nn
from torch.nn import BCELoss, MSELoss
import torch

class VQVAEPerplexity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_index):
        mean = torch.mean(q_index, dim=0)
        return torch.exp(-torch.sum(mean * torch.log(mean + 1e-8)))

class VQVAELoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.bce = BCELoss(reduction="sum")
        self.beta = beta
        self.mse = MSELoss(reduction="sum")
        
    def forward(self, x, recons, z_e, z_q, _):
        L_recons = self.bce(recons, x)
        L_embed = self.mse(z_e.detach(), z_q)
        L_encoder = self.mse(z_e, z_q.detach())
        return L_recons + L_embed + L_encoder * self.beta

class CVQVAECodebook(nn.Module):
    def __init__(self, num_embeddings=320, latent_dim=3):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.latent_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, -1 / self.num_embeddings)
    
    def forward(self, x):
        # x is B x DIM
        # embedding is K x DIM
        distances = (torch.sum(x ** 2, dim=1, keepdim=True) # B x 1
         + torch.sum(self.embedding.weight ** 2, dim=1) # K
         # This will broadcast to B x K
         - 2 * torch.matmul(x, self.embedding.weight.t()) # B x DIM * DIM x K => B x K
        ) # B x K

        codebook_indices = torch.argmin(distances, dim=1).long()
        probs = torch.zeros(x.size(0), self.num_embeddings).to(codebook_indices.device).scatter(1, codebook_indices.unsqueeze(1), 1)
        z_q = self.embedding(codebook_indices)
        return z_q, probs

class MNISTCVQVAE(nn.Module):
    def __init__(self, encoder, decoder, codebook, latent_dim=3, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = codebook

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.fc = nn.Linear(self.embed_dim, self.latent_dim)

    def forward(self, x, c):
        z_e = self.fc(self.encoder(x, c))
        z_q, probs = self.codebook(z_e)
        return self.decoder(z_q, c), z_e, z_q, probs