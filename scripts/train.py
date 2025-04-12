# 1. Importer DatasetLoader, DenoiserUNet, DDPM(DenoiserUNet) 
# 2. Faire la pipeline d'entrainement de DDPM
# 3. Enregistrer le modèle entrainé

# Exécuter ce script depuis la racine du projet avec la commande suivante:
# python -m scripts.train

from preprocessing.dataset import DatasetLoader, download_kaggle_dataset
from model.unet import UNet
from diffusion.diffusion import DDPM
from diffusion.scheduler import make_beta_schedule
import torch





if __name__ == "__main__":
    path = download_kaggle_dataset("brilja/pokemon-mugshots-from-super-mystery-dungeon")
    
    # Créer et configurer le dataloader
    loader = DatasetLoader(
        dataset_path=path,
        img_size=64,
        batch_size=16,
        train_ratio=0.8,
        normalize=True,
    )
    
    # Charger les données
    loader.load_data()
    
    # Obtenir les dataloaders
    train_loader, val_loader, all_loader = loader.get_data()
    
    # Afficher des informations sur les dataloaders
    print(f"Nombre de batchs d'entraînement: {len(train_loader)}")
    print(f"Nombre de batchs de validation: {len(val_loader)}")
    print(f"Taille du batch: {loader.batch_size}")
    print(f"Dimensions des images: {next(iter(train_loader)).shape}")
    print(f"Nombre total d'images: {len(all_loader)}")




    # Créer le modèle UNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")

    channels = 3    #RGB
    n_channels = 64  # Nombre de canaux de base
    ch_mults = (1, 2, 2, 4)  # Multiplicateurs de canaux par niveau
    is_attn = (False, False, True, True)  # Attention par niveau
    n_blocks = 2  # Nombre de blocs résiduels par niveau


    model = UNet(
        image_channels=channels,
        n_channels=n_channels,
        ch_mults=ch_mults,
        is_attn=is_attn,
        n_blocks=n_blocks
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres du modèle: {n_params:,}")


    # Créer le modèle DDPM
    timesteps = 1000
    betas = make_beta_schedule(schedule="linear", timesteps=timesteps)
    ddpm = DDPM(model=model, betas=betas, device=device)
    print(f"Nombre de timesteps: {len(ddpm.betas)}")

    # Entraîner le modèle # Entrainement court pour tester le code
    num_epochs = 3
    lr = 1e-4
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loss = 0
        for i, batch in enumerate(train_loader):
            x_start = batch.to(device)
            loss += ddpm.train_batch(x_start, optimizer)
            # print chariot pour afficher la progression
            print(f"\rEpoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(train_loader)}", end="")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss / len(train_loader)}")

    # Enregistrer le modèle entraîné
    torch.save(ddpm.state_dict(), "ddpm_model.pth")
    print("Modèle entraîné et enregistré sous ddpm_model.pth")