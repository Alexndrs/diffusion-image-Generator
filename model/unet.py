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
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        self.n_heads = n_heads
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)          #normalization layer
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)    #projections for query, key and values 
        self.output = nn.Linear(n_heads * d_k, n_channels)    #final transformation
        self.scale = d_k ** -0.5
        self.d_k = d_k
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t #t is not used, but it's kept in the arguments because for the attention layer function signature to match with ResidualBlock
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)  #reshape to [batch_size, seq, n_heads * d_k]]
        res = self.output(res)          #transform to [batch_size, seq, n_channels]
        res += x                  #skip connection
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res
    
    
class DownBlock(nn.Module):    #This combines ResidualBlock and AttentionBlock . These are used in the first half of U-Net at each resolution.
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res(x, t)
        h = self.attn(h)
        return h
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res(x, t)
        h = self.attn(h)
        return h
    
    

class MiddleBlock(nn.Module):    #It combines a ResidualBlock , AttentionBlock , followed by another ResidualBlock . This block is applied at the lowest resolution of the U-Net.
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res1(x, t)
        h = self.attn(h)
        h = self.res2(h, t)
        return h
    

class Upsample(nn.Module):            #Scale up the feature map by 2×
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        h = self.conv(x)
        return h
    
class Downsample(nn.Module):        #Scale down the feature map by 21​×
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        h = self.conv(x)
        return h
        
    
class UNet(nn.Module):