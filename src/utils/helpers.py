import os
import torch
import numpy as np

def save_checkpoint(state, is_best, save_dir, dataset_name, filename=None):
    """
    Save checkpoint to disk
    
    Args:
        state: Dictionary containing model state
        is_best: Whether this is the best model so far
        save_dir: Directory to save the checkpoint
        dataset_name: Name of the dataset
        filename: Optional custom filename for the checkpoint
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Use custom filename if provided, otherwise create default filename
    if filename is not None:
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(state, checkpoint_path)
        if is_best:
            print(f"Saved best model to {checkpoint_path}")
    else:
        # Save regular checkpoint
        filename = os.path.join(save_dir, f"{dataset_name}_checkpoint_epoch{state['epoch']}.pth")
        torch.save(state, filename)
        
        # Save best model
        if is_best:
            best_filename = os.path.join(save_dir, f"{dataset_name}_best_model.pth")
            torch.save(state, best_filename)
            print(f"Saved best model to {best_filename}")


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve for a given patience.
    """
    def __init__(self, patience=4, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): If True, prints a message for each improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model, epoch, save_dir, dataset_name):
        """
        Call after validation.
        
        Args:
            val_loss (float): Validation loss
            model (nn.Module): Model to save
            epoch (int): Current epoch
            save_dir (str): Directory to save model
            dataset_name (str): Name of dataset
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_dir, dataset_name)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_dir, dataset_name)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model, epoch, save_dir, dataset_name):
        """
        Save model when validation loss decreases.
        
        Args:
            val_loss (float): Validation loss
            model (nn.Module): Model to save
            epoch (int): Current epoch
            save_dir (str): Directory to save model
            dataset_name (str): Name of dataset
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        # Save model
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_val_loss': val_loss,
        }, is_best=True, save_dir=save_dir, dataset_name=dataset_name, filename='early_stopping_checkpoint.pth')
        self.val_loss_min = val_loss


class EMA:
    """
    Exponential Moving Average for model weights
    """
    def __init__(self, model, decay=0.999):
        """
        Args:
            model (nn.Module): Model to apply EMA
            decay (float): EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """
        Update EMA parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """
        Apply EMA parameters to model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """
        Restore original parameters to model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """
        Returns the state of EMA as a dict
        """
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup
        }
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of EMA from a dict
        """
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']
        