# Exécuter depuis : la racine avec
# python -m preprocessing.tests.test

from preprocessing.dataset import DatasetLoader, download_kaggle_dataset



if __name__ == "__main__":
    path = download_kaggle_dataset("brilja/pokemon-mugshots-from-super-mystery-dungeon")
    
    # Créer et configurer le dataloader
    loader = DatasetLoader(
        dataset_path=path,
        img_size=64,
        batch_size=16,
        train_ratio=0.8,
        augmentation=True,
        normalize=True,
    )
    
    # Charger les données
    loader.load_data()
    
    # Visualiser un batch
    loader.visualize_batch(nrow=4, save_path="batch_preview.png")
    
    # Obtenir les dataloaders
    train_loader, val_loader, all_loader = loader.get_data()
    
    # Afficher des informations sur les dataloaders
    print(f"Nombre de batchs d'entraînement: {len(train_loader)}")
    print(f"Nombre de batchs de validation: {len(val_loader)}")
    print(f"Taille du batch: {loader.batch_size}")
    print(f"Dimensions des images: {next(iter(train_loader)).shape}")
