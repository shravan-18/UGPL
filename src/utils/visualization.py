import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Create custom colormap (blue to white to red)
    colors = [(0.0, 0.1, 0.4), (1, 1, 1), (0.7, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, cbar=True)
    
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()

def visualize_uncertainty_map(image, uncertainty_map, patches=None, patch_coords=None, save_path=None):
    """
    Visualize input image, uncertainty map, and selected patches
    
    Args:
        image: Input image tensor [C, H, W]
        uncertainty_map: Uncertainty map tensor [1, H, W]
        patches: Selected patches tensor [K, C, patch_H, patch_W]
        patch_coords: Coordinates of selected patches [K, 4]
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()
    
    if isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.detach().cpu().squeeze().numpy()
    
    if patches is not None and isinstance(patches, torch.Tensor):
        patches = patches.detach().cpu().numpy()
    
    if patch_coords is not None and isinstance(patch_coords, torch.Tensor):
        patch_coords = patch_coords.detach().cpu().numpy()
    
    # Create figure
    n_cols = 3 if patches is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original CT Image')
    axes[0].axis('off')
    
    # Plot uncertainty map
    im = axes[1].imshow(uncertainty_map, cmap='hot', alpha=0.7)
    axes[1].set_title('Uncertainty Map')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot image with patch boxes
    if patches is not None and patch_coords is not None:
        axes[2].imshow(image, cmap='gray')
        axes[2].set_title('Selected Patches')
        axes[2].axis('off')
        
        # Draw patch bounding boxes
        for i, (x1, y1, x2, y2) in enumerate(patch_coords):
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 linewidth=2, edgecolor='r', facecolor='none')
            axes[2].add_patch(rect)
            axes[2].text(x1, y1-5, f"Patch {i+1}", color='r', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved uncertainty visualization to {save_path}")
    
    plt.close()

def visualize_patches(image, patches, patch_coords, local_confidences=None, save_path=None):
    """
    Visualize original image and extracted patches
    
    Args:
        image: Input image tensor [C, H, W]
        patches: Selected patches tensor [K, C, patch_H, patch_W]
        patch_coords: Coordinates of selected patches [K, 4]
        local_confidences: Confidence values for patches [K, 1]
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()
    
    if isinstance(patches, torch.Tensor):
        patches = patches.detach().cpu().squeeze().numpy()
    
    if isinstance(patch_coords, torch.Tensor):
        patch_coords = patch_coords.detach().cpu().numpy()
    
    if local_confidences is not None and isinstance(local_confidences, torch.Tensor):
        local_confidences = local_confidences.detach().cpu().squeeze().numpy()
    
    # Get number of patches
    n_patches = patches.shape[0] if len(patches.shape) > 3 else 1
    
    # Create figure
    fig, axes = plt.subplots(1, n_patches + 1, figsize=((n_patches + 1) * 4, 4))
    
    # Plot original image with patch boxes
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image with Patches')
    axes[0].axis('off')
    
    # Draw patch bounding boxes
    for i, (x1, y1, x2, y2) in enumerate(patch_coords):
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, f"Patch {i+1}", color='r', fontsize=10)
    
    # Plot individual patches
    for i in range(n_patches):
        patch = patches[i] if n_patches > 1 else patches
        
        axes[i+1].imshow(patch, cmap='gray')
        
        if local_confidences is not None:
            conf = local_confidences[i] if n_patches > 1 else local_confidences
            axes[i+1].set_title(f"Patch {i+1}\nConfidence: {conf:.3f}")
        else:
            axes[i+1].set_title(f"Patch {i+1}")
            
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved patch visualization to {save_path}")
    
    plt.close()
    