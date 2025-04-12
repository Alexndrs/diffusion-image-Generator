# Définir les variables globales ici (par ex : batch_size) (voir /preprocessing/README.md)

import torch
from preprocessing.dataset import DatasetLoader, download_kaggle_dataset

DATASET_PATH = download_kaggle_dataset("brilja/pokemon-mugshots-from-super-mystery-dungeon")
# Configuration des paramètres globaux 
IMG_SIZE = 64
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
NORMALIZE = True
AUGMENTATION = True

if __name__ == "__main__":
    # Vérifier si CUDA est disponible et afficher le nombre de GPUs
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")