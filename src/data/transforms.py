from torchvision import transforms

def get_transforms(input_size, mode='train'):
    """
    Get transforms for data augmentation and normalization
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    else:
        return None  # No augmentation for validation/test
    