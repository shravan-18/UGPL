import torch
from torch.utils.data import random_split

from src.data.dataset import CTDataset
from src.data.transforms import get_transforms

def prepare_datasets(dataset_path, dataset_type, input_size, val_split=0.1, test_split=0.2, seed=42):
    """
    Prepare train, validation, and test datasets
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create full dataset
    full_dataset = CTDataset(
        root_dir=dataset_path,
        dataset_type=dataset_type,
        input_size=input_size,
        mode='train',  # Initially set to train mode
        transform=None  # We'll apply transforms later
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - test_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Set the appropriate mode and transforms for each split
    train_dataset.dataset.mode = 'train'
    train_dataset.dataset.transform = get_transforms(input_size, mode='train')
    
    val_dataset.dataset.mode = 'val'
    val_dataset.dataset.transform = None  # No augmentation for validation
    
    test_dataset.dataset.mode = 'test'
    test_dataset.dataset.transform = None  # No augmentation for test
    
    return train_dataset, val_dataset, test_dataset
