# Définir les variables globales ici (par ex : batch_size) (voir /preprocessing/README.md)

import torch

if __name__ == "__main__":
    # Vérifier si CUDA est disponible et afficher le nombre de GPUs
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")