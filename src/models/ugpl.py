import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.global_model import GlobalUncertaintyEstimator
from src.models.patch_extractor import ProgressivePatchExtractor
from src.models.local_model import LocalRefinementNetwork
from src.models.fusion import AdaptiveFusionModule

class UGPL(nn.Module):
    """
    Full Uncertainty-Guided Progressive Learning model
    """
    def __init__(self, num_classes, input_size=256, patch_size=64, num_patches=3, backbone='resnet34', ablation_mode=None):
        super(UGPL, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.ablation_mode = ablation_mode
        
        # Global model for initial classification and uncertainty estimation
        self.global_model = GlobalUncertaintyEstimator(num_classes, backbone=backbone)
        
        # Patch extractor for finding uncertain regions
        self.patch_extractor = ProgressivePatchExtractor(patch_size, num_patches, input_size)
        
        # Local model for detailed patch analysis
        self.local_model = LocalRefinementNetwork(num_classes, patch_size)
        
        # Fusion module for combining global and local predictions
        self.fusion_module = AdaptiveFusionModule(num_classes)
    
    def forward(self, x, return_intermediate=False):
        """
        Forward pass through the complete UGPL model
        
        Args:
            x: Input images of shape [B, C, H, W]
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Global analysis
        global_logits, uncertainty_map, uncertainty_params = self.global_model(x)
        
        # For ablation: If global_only mode, return only global prediction
        if self.ablation_mode == 'global_only':
            return {
                'fused_logits': global_logits,
                'global_logits': global_logits,
                'global_weight': torch.ones(batch_size, 1, device=device),
                'uncertainty_map': uncertainty_map if return_intermediate else None,
            }
        
        # For ablation: If no_uncertainty mode, use uniform uncertainty
        if self.ablation_mode == 'no_uncertainty':
            uncertainty_map = torch.ones_like(uncertainty_map) / uncertainty_map.numel()
        
        # Extract patches from uncertain regions
        patches, patch_coords = self.patch_extractor(x, uncertainty_map)
        
        # For ablation: If fixed_patches mode, extract center patches
        if self.ablation_mode == 'fixed_patches':
            # Create fixed grid of patches
            H, W = x.shape[2], x.shape[3]
            coords = []
            
            # Grid positions (e.g., center and four corners)
            positions = [
                (H // 2, W // 2),  # Center
                (H // 4, W // 4),  # Top-left
                (H // 4, 3 * W // 4),  # Top-right
                (3 * H // 4, W // 4),  # Bottom-left
                (3 * H // 4, 3 * W // 4),  # Bottom-right
            ]
            
            for y_center, x_center in positions[:self.num_patches]:
                half_size = self.patch_size // 2
                x1 = max(0, x_center - half_size)
                y1 = max(0, y_center - half_size)
                x2 = min(W, x_center + half_size)
                y2 = min(H, y_center + half_size)
                coords.append((int(x1), int(y1), int(x2), int(y2)))
            
            # Extract these fixed patches
            patches = torch.zeros(batch_size, self.num_patches, x.shape[1], 
                                 self.patch_size, self.patch_size, device=device)
            patch_coords = torch.zeros(batch_size, self.num_patches, 4, device=device)
            
            for b in range(batch_size):
                for i, (x1, y1, x2, y2) in enumerate(coords):
                    patch = x[b, :, y1:y2, x1:x2]
                    
                    # Handle patches that might be smaller than patch_size
                    if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
                        patch = F.interpolate(
                            patch.unsqueeze(0), 
                            size=(self.patch_size, self.patch_size),
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    
                    patches[b, i] = patch
                    patch_coords[b, i] = torch.tensor([x1, y1, x2, y2])
        
        # Local refinement
        local_logits, local_confidences = self.local_model(patches)
        
        # Fusion
        fused_logits, global_weight = self.fusion_module(
            global_logits, uncertainty_map, local_logits, local_confidences
        )
        
        # Prepare output dictionary
        output = {
            'fused_logits': fused_logits,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'local_confidences': local_confidences,
            'global_weight': global_weight,
        }
        
        # Add intermediate results if requested
        if return_intermediate:
            output.update({
                'uncertainty_map': uncertainty_map,
                'uncertainty_params': uncertainty_params,
                'patches': patches,
                'patch_coords': patch_coords,
            })
        
        return output
    