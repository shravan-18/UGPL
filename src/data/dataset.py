import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt

class CTDataset(Dataset):
    """Dataset for CT images with preprocessing and augmentation"""
    
    def __init__(self, root_dir, dataset_type, input_size=256, mode='train', transform=None):
        """
        Args:
            root_dir: Path to the dataset
            dataset_type: Type of dataset ('kidney', 'lung', 'covid')
            input_size: Size to resize images to
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.mode = mode
        self.input_size = input_size
        self.transform = transform
        self.classes = self._get_classes()
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {mode} samples from {dataset_type} dataset")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_classes(self):
        """Get the classes based on dataset type"""
        if self.dataset_type == 'kidney':
            return ['Normal', 'Cyst', 'Tumor', 'Stone']
        elif self.dataset_type == 'lung':
            return ['benign', 'malignant', 'normal']
        elif self.dataset_type == 'covid':
            return ['covid', 'nonCovid']
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_samples(self):
        """Load all image paths and their corresponding labels"""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory {class_dir} not found!")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dcm')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.class_to_idx[class_name]))
        
        return samples
    
    def _get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        distribution = {cls: 0 for cls in self.classes}
        
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
            
        return distribution
    
    def _load_and_preprocess(self, img_path):
        """Load and preprocess a CT image"""
        try:
            # Load image
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Resize to target size
            image = image.resize((self.input_size, self.input_size), Image.LANCZOS)
            
            # Convert to numpy array
            image_np = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1] range
            image_np = self._normalize_intensity(image_np)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # Add channel dimension
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            return torch.zeros(1, self.input_size, self.input_size)
    
    def _normalize_intensity(self, image, window_center=None, window_width=None):
        """
        Normalize CT image intensity using windowing if provided,
        otherwise use min-max normalization
        """
        if window_center is not None and window_width is not None:
            # Apply CT windowing
            min_val = window_center - window_width // 2
            max_val = window_center + window_width // 2
            image = np.clip(image, min_val, max_val)
        
        # Min-max normalization to [0, 1]
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_and_preprocess(img_path)
        
        # Apply transformations if in training mode
        if self.mode == 'train' and self.transform:
            image = self.transform(image)
            
        return image, label


def visualize_dataset_samples(dataset, num_samples=5, save_path=None):
    """
    Visualize random samples from the dataset
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Convert to numpy for visualization
        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()
        
        # Get class name
        class_name = dataset.dataset.classes[label] if hasattr(dataset, 'dataset') else dataset.classes[label]
        
        # Display image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Class: {class_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset visualization to {save_path}")
    
    plt.show()
    