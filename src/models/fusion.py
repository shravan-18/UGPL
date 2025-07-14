import torch
import torch.nn as nn

class AdaptiveFusionModule(nn.Module):
    """
    Fuses global and local predictions based on confidence
    """
    def __init__(self, num_classes):
        super(AdaptiveFusionModule, self).__init__()
        
        # Network to compute fusion weights
        self.fusion_network = nn.Sequential(
            nn.Linear(num_classes + 1, 64),  # global_logits + global_uncertainty
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Weight between 0 and 1
        )
    
    def forward(self, global_logits, global_uncertainty, local_logits, local_confidences):
        """
        Args:
            global_logits: Tensor of shape [B, num_classes]
            global_uncertainty: Tensor of shape [B, 1, H, W]
            local_logits: Tensor of shape [B, K, num_classes]
            local_confidences: Tensor of shape [B, K, 1]
            
        Returns:
            fused_logits: Tensor of shape [B, num_classes]
        """
        batch_size, num_classes = global_logits.shape
        _, num_patches, _ = local_logits.shape
        
        # Calculate global uncertainty score (mean across spatial dimensions)
        # Make sure we're only taking the mean across the correct dimensions
        # global_uncertainty should be [B, 1, H, W]
        if global_uncertainty.dim() == 4:
            global_uncertainty_score = global_uncertainty.mean(dim=(2, 3))  # [B, 1]
        else:
            # Handle unexpected shape (reshape if needed)
            # If global_uncertainty is [B, C, H, W] where C > 1
            global_uncertainty_score = global_uncertainty.mean(dim=(1, 2, 3), keepdim=True)  # [B, 1]
        
        # Ensure global_uncertainty_score has shape [B, 1]
        if global_uncertainty_score.dim() == 1:
            global_uncertainty_score = global_uncertainty_score.unsqueeze(1)
        
        # Prepare input for fusion network
        fusion_input = torch.cat([global_logits, global_uncertainty_score], dim=1)  # [B, num_classes + 1]
        
        # Calculate global weight
        global_weight = self.fusion_network(fusion_input)  # [B, 1]
        
        # Calculate local weight (1 - global_weight)
        local_weight = 1.0 - global_weight  # [B, 1]
        
        # Apply confidence weighting to local predictions
        weighted_local_logits = local_logits * local_confidences  # [B, K, num_classes]
        
        # Average local predictions across patches
        # Ensure proper handling of dimensions
        local_confidences_sum = local_confidences.sum(dim=1) + 1e-6  # [B, 1]
        
        # Sum weighted local logits across patches dimension
        weighted_local_logits_sum = weighted_local_logits.sum(dim=1)  # [B, num_classes]
        
        # Normalize by sum of confidences
        avg_local_logits = weighted_local_logits_sum / local_confidences_sum  # [B, num_classes]
        
        # Combine global and local predictions
        fused_logits = global_weight * global_logits + local_weight * avg_local_logits
        
        return fused_logits, global_weight
    