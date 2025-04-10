import torch
import torch.nn as nn
from .scheduler import make_beta_schedule


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, betas: torch.Tensor, device: str = "cuda"):
        """
        DDPM : Denoising Diffusion Probabilistic Model
        inputs : 
        - model : UNet (ou autre modèle de diffusion)
        - betas : beta schedule (torch.Tensor) (1D tensor de taille num_timesteps)
        """
        super().__init__()
        self.model = model  # UNet
        self.device = device

        # 1. Variables utiles à partir du scheduler de beta
        self.num_timesteps = len(betas)
        self.betas = betas.to(device)
        self.alphas = 1. - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.sqrt_1m_alpha_bar = torch.sqrt(1. - self.alpha_bar)



    # 2. Noising instantané
    def forward_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """

        input : 
        - x_start : image de départ (batch_size, channels, height, width)
        - t : étape de diffusion (batch_size,)
        - noise : bruit (batch_size, channels, height, width) (même dim que x_start)

        - output : x_t : image bruitée (batch_size, channels, height, width)
        
        Ajoute du bruit à x_start à l'étape t selon la formule :
        x_t = sqrt(alpha_bar_t).x_start + sqrt(1 - alpha_bar_t) * noise
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)


        # self.alpha_bar_sqrt[t] : comme t est un tensor 1D et self.alpha_bar_sqrt est un tensor 1D, self.alpha_bar_sqrt[t] est un tensor 1D de taille batch_size, on doit le reshaper pour qu'il ait la même taille que x_start : pour cela on utilise view(-1, 1, 1, 1) : le -1 signifie que la dimension batch_size est conservée, et les 3 autres dimensions sont mises à 1 
        
        x_t = (
            self.alpha_bar_sqrt[t].view(-1, 1, 1, 1) * x_start +
            self.sqrt_1m_alpha_bar[t].view(-1, 1, 1, 1) * noise
        )
        return x_t


    # 3. Noising successif (optionnel, pour visualisation étape par étape)
    def q_sample_progressive(self, x_start: torch.Tensor):
        """Génère une séquence d'images bruitées étape par étape"""
        ...

    # 4. Calcul de la loss (training)
    def loss_fn(self, x_start: torch.Tensor):
        """Implémente l'algo 1"""
        ...

    # 5. Sampling (inférence)
    @torch.no_grad()
    def sample(self, shape: torch.Size):
        """Implémente l'algo 2"""
        ...

