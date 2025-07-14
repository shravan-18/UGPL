import torch
import torch.nn as nn
import torch.nn.functional as F

class UGPLLoss(nn.Module):
    """
    Multi-component loss function for the UGPL model - Ablation Aware
    """
    def __init__(self, num_classes, weights=None):
        super(UGPLLoss, self).__init__()
        
        # Default loss component weights
        self.weights = {
            'fused': 1.0,
            'global': 0.5,
            'local': 0.5,
            'uncertainty': 0.3,
            'consistency': 0.2,
            'confidence': 0.1,
            'diversity': 0.1,
        }
        
        # Update weights if provided
        if weights is not None:
            self.weights.update(weights)
        
        # Cross-entropy loss for classification
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Number of classes
        self.num_classes = num_classes
    
    def forward(self, model_output, targets):
        """
        Compute the multi-component loss - handles different ablation modes
        
        Args:
            model_output: Dictionary of model outputs
            targets: Ground truth labels [B]
            
        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Dictionary containing individual loss components
        """
        device = targets.device
        
        # Extract outputs from model (with defaults for missing keys)
        fused_logits = model_output.get('fused_logits')
        global_logits = model_output.get('global_logits')
        local_logits = model_output.get('local_logits', None)  # May be None in ablation modes
        local_confidences = model_output.get('local_confidences', None)  # May be None
        
        # Validate required outputs
        if fused_logits is None or global_logits is None:
            raise ValueError("Model must provide 'fused_logits' and 'global_logits'")
        
        # Create one-hot encoded targets
        batch_size = targets.size(0)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # 1. Main classification losses (always computed)
        fused_loss = self.ce_loss(fused_logits, targets)
        global_loss = self.ce_loss(global_logits, targets)
        
        # 2. Local loss (only if local_logits available)
        local_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if local_logits is not None:
            num_patches = local_logits.size(1)
            for i in range(num_patches):
                patch_logits = local_logits[:, i]
                local_loss = local_loss + self.ce_loss(patch_logits, targets)
            local_loss = local_loss / num_patches
        
        # 3. Uncertainty calibration loss (optional)
        uncertainty_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'uncertainty_map' in model_output and model_output['uncertainty_map'] is not None:
            uncertainty_map = model_output['uncertainty_map']
            
            # Global prediction correctness
            global_preds = torch.argmax(global_logits, dim=1)
            correctness_scalar = (global_preds == targets).float()
            
            # Handle uncertainty map dimensions
            if uncertainty_map.dim() == 4:  # [B, C, H, W]
                batch_size, channels, height, width = uncertainty_map.shape
                correctness = correctness_scalar.view(-1, 1, 1, 1).expand(-1, channels, height, width)
            else:  # Handle other dimensions gracefully
                correctness = correctness_scalar
            
            # Uncertainty should be high where prediction is wrong
            uncertainty_target = 1.0 - correctness
            uncertainty_loss = F.mse_loss(uncertainty_map, uncertainty_target)
        
        # 4. Consistency loss (only if local components available)
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if local_logits is not None and local_confidences is not None:
            num_patches = local_logits.size(1)
            for i in range(num_patches):
                patch_logits = local_logits[:, i]
                patch_confidence = local_confidences[:, i]
                
                # KL divergence between global and local predictions
                global_probs = F.softmax(global_logits, dim=1)
                local_probs = F.softmax(patch_logits, dim=1)
                
                kl_div = F.kl_div(
                    torch.log(local_probs + 1e-10),
                    global_probs,
                    reduction='none'
                ).sum(dim=1)
                
                # Weight by patch confidence
                weighted_kl = kl_div * patch_confidence.squeeze()
                consistency_loss = consistency_loss + weighted_kl.mean()
            
            consistency_loss = consistency_loss / num_patches
        
        # 5. Confidence regularization (only if local components available)
        confidence_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if local_logits is not None and local_confidences is not None:
            num_patches = local_logits.size(1)
            for i in range(num_patches):
                patch_logits = local_logits[:, i]
                patch_confidence = local_confidences[:, i]
                
                # Local prediction correctness
                patch_preds = torch.argmax(patch_logits, dim=1)
                patch_correctness = (patch_preds == targets).float().unsqueeze(1)
                
                # Confidence should match correctness
                confidence_loss = confidence_loss + F.mse_loss(patch_confidence, patch_correctness)
            
            confidence_loss = confidence_loss / num_patches
        
        # 6. Patch diversity loss (only if local_logits available and multiple patches)
        diversity_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if local_logits is not None:
            num_patches = local_logits.size(1)
            if num_patches > 1:
                for i in range(num_patches):
                    for j in range(i+1, num_patches):
                        # Encourage different predictions between patches
                        logits_i = F.softmax(local_logits[:, i], dim=1)
                        logits_j = F.softmax(local_logits[:, j], dim=1)
                        
                        # Cosine similarity (lower is more diverse)
                        similarity = F.cosine_similarity(logits_i, logits_j, dim=1)
                        diversity_loss = diversity_loss + similarity.mean()
                
                # Normalize by number of patch pairs
                num_pairs = (num_patches * (num_patches - 1)) // 2
                diversity_loss = diversity_loss / num_pairs
        
        # Combine losses with weights (only non-zero losses contribute)
        total_loss = (
            self.weights['fused'] * fused_loss +
            self.weights['global'] * global_loss +
            self.weights['local'] * local_loss +
            self.weights['uncertainty'] * uncertainty_loss +
            self.weights['consistency'] * consistency_loss +
            self.weights['confidence'] * confidence_loss +
            self.weights['diversity'] * diversity_loss
        )
        
        # Create dictionary of individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'fused': fused_loss.item(),
            'global': global_loss.item(),
            'local': local_loss.item(),
            'uncertainty': uncertainty_loss.item(),
            'consistency': consistency_loss.item(),
            'confidence': confidence_loss.item(),
            'diversity': diversity_loss.item(),
        }
        
        return total_loss, loss_dict

class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning loss for uncertainty estimation
    Based on the paper: Evidential Deep Learning to Quantify Classification Uncertainty
    """
    def __init__(self, num_classes, lambda_reg=0.1):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
    
    def forward(self, alpha, target):
        """
        Compute the evidential loss
        
        Args:
            alpha: Dirichlet concentration parameters [B, C, H, W]
            target: Ground truth labels [B]
            
        Returns:
            loss: Evidential loss
        """
        # Convert target to one-hot encoding
        target_oh = F.one_hot(target, self.num_classes).float()
        
        # Reshape alpha to match target if needed
        if len(alpha.shape) > 2:
            # If alpha has spatial dimensions, average over spatial dimensions
            alpha = alpha.mean(dim=(2, 3))
        
        # Compute precision (sum of alpha)
        S = alpha.sum(dim=1, keepdim=True)
        
        # Compute mean (expected probability)
        m = alpha / S
        
        # Compute squared error loss
        A = torch.sum(target_oh * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        # Compute regularization term (KL divergence to encourage low precision for wrong predictions)
        reg = self.lambda_reg * torch.sum((1 - target_oh) * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        
        # Total loss
        loss = A + reg
        
        return loss.mean()
    