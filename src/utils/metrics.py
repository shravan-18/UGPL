import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, 
    roc_auc_score, precision_recall_fscore_support, auc, roc_curve
)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Numpy array of predicted class indices
        targets: Numpy array of true class indices
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    # Number of classes
    num_classes = len(np.unique(targets))
    
    # Basic metrics
    accuracy = (predictions == targets).mean()
    
    # Confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)
    
    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # Balanced accuracy
    balanced_acc = per_class_acc.mean()
    
    # F1 scores
    f1_micro = f1_score(targets, predictions, average='micro')
    f1_macro = f1_score(targets, predictions, average='macro')
    f1_weighted = f1_score(targets, predictions, average='weighted')
    f1_per_class = f1_score(targets, predictions, average=None)
    
    # Precision, recall per class
    precision, recall, _, _ = precision_recall_fscore_support(targets, predictions, average=None)
    
    # ROC AUC score - need probabilities for this, so we'll calculate it separately
    # We'll just define placeholders here
    roc_auc_macro = None
    roc_auc_weighted = None
    roc_auc_per_class = None
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': conf_matrix,
        'per_class_accuracy': per_class_acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'roc_auc_per_class': roc_auc_per_class
    }
    
    return metrics

def calculate_roc_auc_metrics(probabilities, targets):
    """
    Calculate ROC AUC metrics
    
    Args:
        probabilities: Numpy array of predicted probabilities [n_samples, n_classes]
        targets: Numpy array of true class indices
        
    Returns:
        metrics: Dictionary containing ROC AUC metrics
    """
    num_classes = probabilities.shape[1]
    
    # One-hot encode targets for multi-class ROC AUC
    targets_one_hot = np.eye(num_classes)[targets]
    
    # Compute ROC AUC per class
    roc_auc_per_class = []
    for i in range(num_classes):
        if len(np.unique(targets_one_hot[:, i])) > 1:  # Check if class exists in targets
            try:
                roc_auc = roc_auc_score(targets_one_hot[:, i], probabilities[:, i])
                roc_auc_per_class.append(roc_auc)
            except ValueError:
                # This happens when a class is not present in the targets
                roc_auc_per_class.append(np.nan)
        else:
            roc_auc_per_class.append(np.nan)
    
    # Compute macro average ROC AUC (average across classes)
    valid_auc = [auc for auc in roc_auc_per_class if not np.isnan(auc)]
    roc_auc_macro = np.mean(valid_auc) if valid_auc else np.nan
    
    # Compute weighted average ROC AUC (weighted by class support)
    class_support = np.sum(targets_one_hot, axis=0)
    weights = class_support / np.sum(class_support)
    valid_indices = ~np.isnan(roc_auc_per_class)
    roc_auc_weighted = np.sum(weights[valid_indices] * np.array(roc_auc_per_class)[valid_indices]) / np.sum(weights[valid_indices]) if np.any(valid_indices) else np.nan
    
    return {
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'roc_auc_per_class': roc_auc_per_class
    }
