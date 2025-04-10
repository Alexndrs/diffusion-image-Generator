# Exécuter depuis : la racine avec
# python -m diffusion.tests.test

import torch
import torchvision.transforms as T
from PIL import Image
import os

from diffusion.diffusion import DDPM
from diffusion.scheduler import make_beta_schedule

def test_instant_forward_noise():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger et transformer l'image
    img_path = "./diffusion/tests/images/imageTest.png"
    image = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        # T.Resize((64, 64)),
        T.Resize((image.size[1] // 4, image.size[0] // 4)),  # Réduire la taille de l'image
        T.ToTensor(),  # convert to [0,1]
        T.Normalize([0.5]*3, [0.5]*3),  # normalize to [-1, 1]
    ])
    x_start = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 64, 64)

    # Créer le scheduler beta et le DDPM (model=None car inutile ici)
    betas = make_beta_schedule(schedule="linear", timesteps=1000)
    ddpm = DDPM(model=None, betas=betas, device=device)

    # Noiser à différents instants t
    t_values = [10, 100, 250, 500, 750, 999]
    for t in t_values:
        t_tensor = torch.tensor([t], device=device)
        noisy = ddpm.forward_noise(x_start, t_tensor)

        # Dénormaliser pour affichage
        unnormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        img_noisy = unnormalize(noisy.squeeze(0).cpu().clamp(-1, 1))
        img_pil = T.ToPILImage()(img_noisy)

        save_path = f"./diffusion/tests/images/instant_noising/imageTest_t{t}.png"
        img_pil.save(save_path)

    print("✅ Test de noising instantané terminé.")

if __name__ == "__main__":
    test_instant_forward_noise()
