import os
import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.training.loss import UGPLLoss
from src.utils.metrics import calculate_metrics, calculate_roc_auc_metrics, AverageMeter
from src.utils.helpers import save_checkpoint

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                epochs=100, dataset_name='kidney', save_dir='checkpoints',
                use_ema=True, ema_decay=0.999, early_stopping_patience=4):
    """
    Train the UGPL model
    
    Args:
        model: UGPL model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs
        dataset_name: Name of the dataset
        save_dir: Directory to save checkpoints
        use_ema: Whether to use EMA
        ema_decay: EMA decay rate
        early_stopping_patience: Number of epochs to wait before early stopping
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize loss function
    criterion = UGPLLoss(model.num_classes)
    
    # Initialize EMA if requested
    ema = None
    if use_ema:
        from src.utils.helpers import EMA
        ema = EMA(model, decay=ema_decay)
    
    # Initialize early stopping
    from src.utils.helpers import EarlyStopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    # Initialize best validation accuracy
    best_val_acc = 0.0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1_macro': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_macro': [],
        'val_roc_auc_macro': [],
    }
    
    # Start training
    print(f"Starting training for {epochs} epochs on {dataset_name} dataset")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Update EMA parameters
        if use_ema:
            ema.update()
            ema.apply_shadow()  # Apply EMA for validation
        
        # Evaluate on validation set
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Restore original parameters if using EMA
        if use_ema:
            ema.restore()
        
        # Update learning rate
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_roc_auc_macro'].append(val_metrics.get('roc_auc_macro', float('nan')))
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_metrics['f1_macro']:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")
        
        # Save checkpoint if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'ema_state_dict': ema.state_dict() if use_ema else None,
            }, is_best=True, save_dir=save_dir, dataset_name=dataset_name)
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'ema_state_dict': ema.state_dict() if use_ema else None,
            }, is_best=False, save_dir=save_dir, dataset_name=dataset_name)
        
        # Check for early stopping
        if early_stopping(val_loss, model, epoch, save_dir, dataset_name):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Apply EMA for final model if using EMA
    if use_ema:
        ema.apply_shadow()
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training complete in {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plot_training_history(history, save_dir, dataset_name)
    
    return history

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: UGPL model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss for the epoch
        avg_acc: Average accuracy for the epoch
        metrics: Dictionary containing detailed metrics
    """
    # Set model to training mode
    model.train()
    
    # Initialize metrics
    losses = AverageMeter('Loss', ':.4f')
    accs = AverageMeter('Acc', ':.4f')
    component_losses = {
        'fused': AverageMeter('Fused', ':.4f'),
        'global': AverageMeter('Global', ':.4f'),
        'local': AverageMeter('Local', ':.4f'),
        'uncertainty': AverageMeter('Uncertainty', ':.4f'),
        'consistency': AverageMeter('Consistency', ':.4f'),
        'confidence': AverageMeter('Confidence', ':.4f'),
        'diversity': AverageMeter('Diversity', ':.4f'),
    }
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    # Lists to store predictions and targets for metric calculation
    all_preds = []
    all_targets = []
    
    for i, (images, targets) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images, return_intermediate=(i == 0))  # Return intermediate results for first batch only
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        for k, v in loss_dict.items():
            if k in component_losses:
                component_losses[k].update(v, images.size(0))
        
        # Compute accuracy
        preds = torch.argmax(outputs['fused_logits'], dim=1)
        acc = (preds == targets).float().mean().item()
        accs.update(acc, images.size(0))
        
        # Store predictions and targets for metric calculation
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'acc': f"{accs.avg:.4f}",
            'g_loss': f"{component_losses['global'].avg:.4f}",
            'l_loss': f"{component_losses['local'].avg:.4f}",
        })
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate detailed metrics
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    metrics.update({
        'loss': losses.avg,
        'acc': accs.avg,
    })
    
    # Add component losses to metrics
    for k, meter in component_losses.items():
        metrics[f"{k}_loss"] = meter.avg
    
    return losses.avg, accs.avg, metrics

def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set
    
    Args:
        model: UGPL model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        avg_loss: Average loss for the validation set
        avg_acc: Average accuracy for the validation set
        metrics: Dictionary containing detailed metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    losses = AverageMeter('Loss', ':.4f')
    accs = AverageMeter('Acc', ':.4f')
    
    # Use tqdm for progress bar
    pbar = tqdm(val_loader, desc="Validation")
    
    # Lists to store predictions, targets, and probabilities for metric calculation
    all_preds = []
    all_targets = []
    all_probs = []  # Store probabilities for ROC AUC calculation
    
    with torch.no_grad():
        for images, targets in pbar:
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss, _ = criterion(outputs, targets)
            
            # Get probabilities for ROC AUC calculation
            probs = torch.softmax(outputs['fused_logits'], dim=1)
            
            # Compute accuracy
            preds = torch.argmax(outputs['fused_logits'], dim=1)
            acc = (preds == targets).float().mean().item()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))
            
            # Store predictions, targets, and probabilities for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc': f"{accs.avg:.4f}",
            })
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate detailed metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    # Calculate ROC AUC metrics
    roc_auc_metrics = calculate_roc_auc_metrics(all_probs, all_targets)
    metrics.update(roc_auc_metrics)
    
    # Update metrics with loss and accuracy
    metrics.update({
        'loss': losses.avg,
        'acc': accs.avg,
    })
    
    return losses.avg, accs.avg, metrics

def plot_training_history(history, save_dir, dataset_name):
    """
    Plot training and validation loss/accuracy/f1/roc_auc
    
    Args:
        history: Dictionary containing training history
        save_dir: Directory to save the plot
        dataset_name: Name of the dataset
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axs[0, 0].plot(history['train_loss'], label='Train')
    axs[0, 0].plot(history['val_loss'], label='Validation')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Loss vs. Epoch')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axs[0, 1].plot(history['train_acc'], label='Train')
    axs[0, 1].plot(history['val_acc'], label='Validation')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Accuracy vs. Epoch')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 score
    axs[1, 0].plot(history['train_f1_macro'], label='Train')
    axs[1, 0].plot(history['val_f1_macro'], label='Validation')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('F1 Score (Macro)')
    axs[1, 0].set_title('F1 Score vs. Epoch')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot ROC AUC
    if 'val_roc_auc_macro' in history:
        # Filter out NaN values
        val_roc_auc = [x for x in history['val_roc_auc_macro'] if not np.isnan(x)]
        epochs = list(range(len(val_roc_auc)))
        
        if val_roc_auc:  # Only plot if we have valid values
            axs[1, 1].plot(epochs, val_roc_auc, label='Validation')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('ROC AUC (Macro)')
            axs[1, 1].set_title('ROC AUC vs. Epoch')
            axs[1, 1].legend()
            axs[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_training_history.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    