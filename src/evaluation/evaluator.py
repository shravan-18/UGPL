import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.metrics import AverageMeter, calculate_metrics
from src.utils.visualization import plot_confusion_matrix

def evaluate_model(model, test_loader, device, dataset_name, save_dir='results'):
    """
    Evaluate the model on the test set
    
    Args:
        model: UGPL model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        dataset_name: Name of the dataset
        save_dir: Directory to save results
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    accs = AverageMeter('Acc', ':.4f')
    
    # Lists to store predictions, targets, and confidences
    all_global_preds = []
    all_local_preds = []
    all_fused_preds = []
    all_targets = []
    all_global_probs = []
    all_fused_probs = []
    all_global_weights = []
    
    print(f"Evaluating model on {dataset_name} test set...")
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluation"):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images, return_intermediate=True)
            
            # Get predictions
            global_logits = outputs['global_logits']
            fused_logits = outputs['fused_logits']
            global_weight = outputs['global_weight']
            
            global_probs = torch.softmax(global_logits, dim=1)
            fused_probs = torch.softmax(fused_logits, dim=1)
            
            global_preds = torch.argmax(global_logits, dim=1)
            fused_preds = torch.argmax(fused_logits, dim=1)
            
            # Local predictions (average across patches)
            local_logits = outputs['local_logits']  # [B, K, num_classes]
            local_confidences = outputs['local_confidences']  # [B, K, 1]
            
            # Weighted average of local predictions
            weighted_local_logits = local_logits * local_confidences
            avg_local_logits = weighted_local_logits.sum(dim=1) / (local_confidences.sum(dim=1) + 1e-6)
            local_preds = torch.argmax(avg_local_logits, dim=1)
            
            # Compute accuracy
            acc = (fused_preds == targets).float().mean().item()
            accs.update(acc, images.size(0))
            
            # Store predictions and targets
            all_global_preds.extend(global_preds.cpu().numpy())
            all_local_preds.extend(local_preds.cpu().numpy())
            all_fused_preds.extend(fused_preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_global_probs.extend(global_probs.cpu().numpy())
            all_fused_probs.extend(fused_probs.cpu().numpy())
            all_global_weights.extend(global_weight.cpu().numpy())
    
    # Convert to numpy arrays
    all_global_preds = np.array(all_global_preds)
    all_local_preds = np.array(all_local_preds)
    all_fused_preds = np.array(all_fused_preds)
    all_targets = np.array(all_targets)
    all_global_probs = np.array(all_global_probs)
    all_fused_probs = np.array(all_fused_probs)
    all_global_weights = np.array(all_global_weights)
    
    # Calculate metrics for each prediction type
    global_metrics = calculate_metrics(all_global_preds, all_targets)
    local_metrics = calculate_metrics(all_local_preds, all_targets)
    fused_metrics = calculate_metrics(all_fused_preds, all_targets)
    
    # Get class names
    if dataset_name == 'kidney':
        class_names = ['Normal', 'Cyst', 'Tumor', 'Stone']
    elif dataset_name == 'lung':
        class_names = ['Benign', 'Malignant', 'Normal']
    elif dataset_name == 'covid':
        class_names = ['COVID', 'Non-COVID']
    else:
        class_names = [f"Class {i}" for i in range(len(np.unique(all_targets)))]
    
    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Global Model - Accuracy: {global_metrics['accuracy']:.4f}, F1 (Macro): {global_metrics['f1_macro']:.4f}")
    print(f"Local Model - Accuracy: {local_metrics['accuracy']:.4f}, F1 (Macro): {local_metrics['f1_macro']:.4f}")
    print(f"Fused Model - Accuracy: {fused_metrics['accuracy']:.4f}, F1 (Macro): {fused_metrics['f1_macro']:.4f}")
    
    # Generate classification report
    print("\nClassification Report (Fused Model):")
    print(classification_report(all_targets, all_fused_preds, target_names=class_names))
    
    # Plot confusion matrices
    plot_confusion_matrix(
        global_metrics['confusion_matrix'], 
        class_names, 
        save_path=os.path.join(save_dir, f"{dataset_name}_global_confusion_matrix.pdf")
    )
    
    plot_confusion_matrix(
        fused_metrics['confusion_matrix'], 
        class_names, 
        save_path=os.path.join(save_dir, f"{dataset_name}_fused_confusion_matrix.pdf")
    )
    
    # Plot global vs. fused accuracy comparison
    plt.figure(figsize=(10, 6))
    
    metrics_names = ['Accuracy', 'Balanced Accuracy', 'F1 (Macro)']
    global_values = [global_metrics['accuracy'], global_metrics['balanced_accuracy'], global_metrics['f1_macro']]
    local_values = [local_metrics['accuracy'], local_metrics['balanced_accuracy'], local_metrics['f1_macro']]
    fused_values = [fused_metrics['accuracy'], fused_metrics['balanced_accuracy'], fused_metrics['f1_macro']]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    plt.bar(x - width, global_values, width, label='Global Model', color='#3366CC')
    plt.bar(x, local_values, width, label='Local Model', color='#FF9900')
    plt.bar(x + width, fused_values, width, label='Fused Model', color='#339933')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_performance_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot global weight distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(all_global_weights.flatten(), bins=20, kde=True, color='#5DA5DA')
    plt.xlabel('Global Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Global Weights')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_global_weight_distribution.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combine all metrics
    metrics = {
        'global': global_metrics,
        'local': local_metrics,
        'fused': fused_metrics,
        'overall_accuracy': fused_metrics['accuracy'],
    }
    
    return metrics

def analyze_error_cases(model, test_loader, device, dataset_name, save_dir='results', num_samples=5):
    """
    Analyze error cases
    
    Args:
        model: UGPL model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        dataset_name: Name of the dataset
        save_dir: Directory to save results
        num_samples: Number of error samples to analyze
        
    Returns:
        error_cases: List of error cases with details
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get class names
    if dataset_name == 'kidney':
        class_names = ['Normal', 'Cyst', 'Tumor', 'Stone']
    elif dataset_name == 'lung':
        class_names = ['Benign', 'Malignant', 'Normal']
    elif dataset_name == 'covid':
        class_names = ['COVID', 'Non-COVID']
    else:
        class_names = [f"Class {i}" for i in range(model.num_classes)]
    
    # List to store error cases
    error_cases = []
    
    # Find error cases
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            # Skip if we have enough error cases
            if len(error_cases) >= num_samples:
                break
                
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images, return_intermediate=True)
            
            # Get predictions
            fused_logits = outputs['fused_logits']
            fused_preds = torch.argmax(fused_logits, dim=1)
            
            # Find errors
            errors = (fused_preds != targets).nonzero(as_tuple=True)[0]
            
            # Process each error
            for err_idx in errors:
                # Skip if we have enough error cases
                if len(error_cases) >= num_samples:
                    break
                    
                # Get error details
                image = images[err_idx]
                true_class = targets[err_idx].item()
                pred_class = fused_preds[err_idx].item()
                
                # Get component outputs
                global_logits = outputs['global_logits'][err_idx]
                global_pred = torch.argmax(global_logits).item()
                
                local_logits = outputs['local_logits'][err_idx]  # [K, num_classes]
                local_confidences = outputs['local_confidences'][err_idx]  # [K, 1]
                local_preds = torch.argmax(local_logits, dim=1).cpu().numpy()
                
                global_weight = outputs['global_weight'][err_idx].item()
                
                uncertainty_map = outputs['uncertainty_map'][err_idx] if 'uncertainty_map' in outputs else None
                patches = outputs['patches'][err_idx] if 'patches' in outputs else None
                patch_coords = outputs['patch_coords'][err_idx] if 'patch_coords' in outputs else None
                
                # Add to error cases
                error_cases.append({
                    'image': image.cpu(),
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'global_pred': global_pred,
                    'local_preds': local_preds,
                    'global_weight': global_weight,
                    'uncertainty_map': uncertainty_map.cpu() if uncertainty_map is not None else None,
                    'patches': patches.cpu() if patches is not None else None,
                    'patch_coords': patch_coords.cpu() if patch_coords is not None else None,
                    'local_confidences': local_confidences.cpu(),
                })
    
    # Visualize error cases
    for i, case in enumerate(error_cases):
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Define grid
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(case['image'].squeeze(), cmap='gray')
        ax1.set_title(f"True: {class_names[case['true_class']]}, Pred: {class_names[case['pred_class']]}")
        ax1.axis('off')
        
        # Uncertainty map
        if case['uncertainty_map'] is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            im = ax2.imshow(case['uncertainty_map'].squeeze(), cmap='hot')
            ax2.set_title("Uncertainty Map")
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Image with patch boxes
        if case['patches'] is not None and case['patch_coords'] is not None:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(case['image'].squeeze(), cmap='gray')
            ax3.set_title("Selected Patches")
            ax3.axis('off')
            
            # Draw patch bounding boxes
            for j, (x1, y1, x2, y2) in enumerate(case['patch_coords']):
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='r', facecolor='none')
                ax3.add_patch(rect)
                ax3.text(x1, y1-5, f"Patch {j+1}", color='r', fontsize=10)
        
        # Patches
        if case['patches'] is not None:
            # Determine number of patches
            n_patches = case['patches'].shape[0]
            
            for j in range(n_patches):
                ax = fig.add_subplot(gs[1, j])
                ax.imshow(case['patches'][j].squeeze(), cmap='gray')
                
                local_conf = case['local_confidences'][j].item()
                local_pred = case['local_preds'][j]
                
                ax.set_title(f"Patch {j+1}\nPred: {class_names[local_pred]}\nConf: {local_conf:.3f}")
                ax.axis('off')
        
        # Model decision details
        ax_text = fig.add_subplot(gs[1, -1])
        ax_text.axis('off')
        
        # Prepare text info
        text = f"Error Analysis:\n\n"
        text += f"True Class: {class_names[case['true_class']]}\n"
        text += f"Predicted Class: {class_names[case['pred_class']]}\n\n"
        text += f"Global Prediction: {class_names[case['global_pred']]}\n"
        text += f"Local Predictions:\n"
        
        for j, pred in enumerate(case['local_preds']):
            conf = case['local_confidences'][j].item()
            text += f"  - Patch {j+1}: {class_names[pred]} (Conf: {conf:.3f})\n"
        
        text += f"\nGlobal Weight: {case['global_weight']:.3f}\n"
        text += f"Local Weight: {1.0 - case['global_weight']:.3f}\n"
        
        ax_text.text(0, 0.5, text, fontsize=12, va='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_error_case_{i+1}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return error_cases
