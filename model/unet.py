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
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(n_groups, in_channels)    #groupe normalisation plutôt que batch nor
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
            
        #embedding de t
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))       #first convolution
        h += self.time_emb(self.time_act(t))[:, :, None, None] #ajout du time embedding
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))     #second convolution
        return h + self.shortcut(x)



class AttentionBlock(nn.Module):
    
    
    
    
class DownBlock(nn.Module):
    
    
    
class UpBlock(nn.Module):
    
    
    
class MiddleBlock(nn.Module):
    
    

class Upsample(nn.Module):
    
    
    
class Downsample(nn.Module):
    
    
    
class UNet(nn.Module):