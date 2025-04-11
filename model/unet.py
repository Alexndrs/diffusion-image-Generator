from typing import Optional, Tuple, Union, List
import torch
import math
from torch import nn


#activation function : swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.activation = Swish()
        self.n_channels = n_channels
        self.linear1 = nn.Linear(self.n_channels//4, self.n_channels)      # n//4 en entrée car on prendra un vecteur d'embedding de t générer par un sinusoidal encoding
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.n_channels // 8      #half dim par rapport à l'entrée du MLP
        facteur_exp = math.log(10_000) / (half_dim - 1)   #10000 est une constante choisie arbitrairement
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -facteur_exp)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        #passage dans le MLP
        emb = self.linear1(emb)
        emb = self.activation(emb)
        emb = self.linear2(emb)
        return emb


class ResidualBlock(nn.Module):
    