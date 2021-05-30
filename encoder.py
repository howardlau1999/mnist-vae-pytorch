from torch import nn
import torch

class MLPEncoder(nn.Module):
    def __init__(self, hidden_dim_1=512, hidden_dim_2=256, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(28 * 28 + self.num_classes, self.hidden_dim_1),
            nn.ReLU(),

            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
        )

    def forward(self, x, c):
        xc = torch.cat([x.reshape(-1, 28 * 28), c], dim=1)
        hidden = self.fc(xc)
        return hidden

class ConvEncoder(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7), 
            nn.MaxPool2d(kernel_size=2), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), # B x 16 x 11 x 11

            nn.Conv2d(16, 32, kernel_size=5), 
            nn.BatchNorm2d(32),
            nn.ReLU() # B x 32 x 7 x 7
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7 + self.num_classes, 256), 
            nn.ReLU(),

            nn.Linear(256, 512), 
            nn.ReLU(),

            nn.Linear(512, 512), 
            nn.ReLU()
        )
    
    def forward(self, x, c):
        x = self.conv(x)
        x = x.reshape(-1, 32 * 7 * 7)
        xc = torch.cat([x, c], dim=1) # B x (D + C)
        hidden = self.fc(xc)
        return hidden