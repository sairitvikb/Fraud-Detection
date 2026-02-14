import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class GANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, output_dim: int = 784, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class GANDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 2, output_dim: int = 784):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def generate(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z_with_labels = torch.cat([z, F.one_hot(labels, self.num_classes).float()], dim=1)
        return self.generator(z_with_labels)
    
    def discriminate(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x_with_labels = torch.cat([x, F.one_hot(labels, self.num_classes).float()], dim=1)
        return self.discriminator(x_with_labels)


class WassersteinGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, output_dim: int = 784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class WassersteinGANDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DiffusionModel(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.model = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t.float().unsqueeze(-1))
        x_with_time = torch.cat([x, t_embed], dim=1)
        return self.model(x_with_time)
