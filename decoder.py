from torch import nn
import torch

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim_1=512, hidden_dim_2=256, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, self.hidden_dim_1),
            nn.ReLU(),

            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            
            nn.Linear(self.hidden_dim_2, 28 * 28),
            nn.Sigmoid(),
        )
    
    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.fc(zc).reshape(-1, 1, 28, 28)

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 32 * 14 * 14), 
            nn.ReLU()
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 7), 
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 7)
        )

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        hidden = self.fc(zc)
        return self.sigmoid(self.up_conv(hidden.reshape(-1, 32, 14, 14)))