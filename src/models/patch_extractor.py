import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressivePatchExtractor(nn.Module):
    """
    Extracts patches from the most uncertain regions of the image
    """
    def __init__(self, patch_size=64, num_patches=3, input_size=256):
        super(ProgressivePatchExtractor, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.input_size = input_size
    
    def forward(self, images, uncertainty_maps):
        """
        Extract patches from uncertain regions
        
        Args:
            images: Tensor of shape [B, C, H, W]
            uncertainty_maps: Tensor of shape [B, 1, H, W]
            
        Returns:
            patches: Tensor of shape [B, num_patches, C, patch_size, patch_size]
            patch_coords: Tensor of shape [B, num_patches, 4] (x1, y1, x2, y2)
        """
        batch_size, channels, height, width = images.shape
        device = images.device
        
        # Initialize outputs
        patches = torch.zeros(batch_size, self.num_patches, channels, 
                            self.patch_size, self.patch_size, device=device)
        patch_coords = torch.zeros(batch_size, self.num_patches, 4, device=device)
        
        # Process each image in the batch
        for b in range(batch_size):
            image = images[b]
            uncertainty_map = uncertainty_maps[b, 0]  # [H, W]
            
            # Find top-k uncertain regions with non-maximum suppression
            coords = self._select_patches(uncertainty_map)
            
            # Extract patches for this image
            for i, (x1, y1, x2, y2) in enumerate(coords):
                if i >= self.num_patches:
                    break
                    
                # Extract patch
                patch = image[:, y1:y2, x1:x2]
                
                # Always resize the patch to the standard patch size
                patch = F.interpolate(
                    patch.unsqueeze(0), 
                    size=(self.patch_size, self.patch_size),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                patches[b, i] = patch
                patch_coords[b, i] = torch.tensor([x1, y1, x2, y2])
        
        return patches, patch_coords
    
    def _select_patches(self, uncertainty_map):
        """
        Select non-overlapping patches with highest uncertainty
        
        Args:
            uncertainty_map: Tensor of shape [H, W]
            
        Returns:
            List of (x1, y1, x2, y2) coordinates for patches
        """
        H, W = uncertainty_map.shape
        device = uncertainty_map.device
        
        # Ensure patch size is not larger than image dimensions
        effective_patch_size = min(self.patch_size, H // 2, W // 2)
        half_size = effective_patch_size // 2
        
        # Ensure map is on CPU for numpy operations
        uncertainty_map_np = uncertainty_map.detach().cpu().numpy()
        
        # List to store selected patch coordinates
        selected_coords = []
        
        # Create a mask to keep track of selected regions
        mask = np.zeros_like(uncertainty_map_np, dtype=bool)
        
        # Keep selecting patches until we have enough or can't find more
        while len(selected_coords) < self.num_patches:
            # Skip if the entire map is masked
            if mask.all():
                break
                
            # Find the location with highest uncertainty that isn't masked
            masked_map = np.ma.array(uncertainty_map_np, mask=mask)
            max_loc = np.unravel_index(np.ma.argmax(masked_map), masked_map.shape)
            y_center, x_center = max_loc
            
            # Calculate patch boundaries
            x1 = max(0, x_center - half_size)
            y1 = max(0, y_center - half_size)
            x2 = min(W, x_center + half_size)
            y2 = min(H, y_center + half_size)
            
            # Adjust to maintain patch size
            if x2 - x1 < effective_patch_size:
                if x1 == 0:
                    x2 = min(W, x1 + effective_patch_size)
                else:
                    x1 = max(0, x2 - effective_patch_size)
            
            if y2 - y1 < effective_patch_size:
                if y1 == 0:
                    y2 = min(H, y1 + effective_patch_size)
                else:
                    y1 = max(0, y2 - effective_patch_size)
            
            # Add to selected patches
            selected_coords.append((int(x1), int(y1), int(x2), int(y2)))
            
            # Update mask to include this patch and a small surrounding area (to prevent too close patches)
            margin = max(effective_patch_size // 4, 1)  # Add margin to reduce overlap
            mask_x1 = max(0, x1 - margin)
            mask_y1 = max(0, y1 - margin)
            mask_x2 = min(W, x2 + margin)
            mask_y2 = min(H, y2 + margin)
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = True
        
        # If we couldn't find enough patches, add random ones
        # Check if we have valid dimensions for random selection
        can_select_random = (2 * half_size < W) and (2 * half_size < H)
        
        while len(selected_coords) < self.num_patches and can_select_random:
            # Random center point with bounds checking
            try:
                x_center = np.random.randint(half_size, W - half_size)
                y_center = np.random.randint(half_size, H - half_size)
                
                # Calculate patch boundaries
                x1 = max(0, x_center - half_size)
                y1 = max(0, y_center - half_size)
                x2 = min(W, x_center + half_size)
                y2 = min(H, y_center + half_size)
                
                # Check if this patch overlaps significantly with existing ones
                overlap = False
                for x1_s, y1_s, x2_s, y2_s in selected_coords:
                    # Calculate IoU
                    x_left = max(x1, x1_s)
                    y_top = max(y1, y1_s)
                    x_right = min(x2, x2_s)
                    y_bottom = min(y2, y2_s)
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (x2_s - x1_s) * (y2_s - y1_s)
                        union = area1 + area2 - intersection
                        iou = intersection / union
                        
                        if iou > 0.3:  # Overlap threshold
                            overlap = True
                            break
                
                if not overlap:
                    selected_coords.append((int(x1), int(y1), int(x2), int(y2)))
            except ValueError:
                # If we hit a ValueError here, break out of the loop
                print(f"Warning: Could not select random patches. Image too small for patch size {self.patch_size}")
                break
        
        # If we still don't have enough patches, use the same patch multiple times as fallback
        while len(selected_coords) < self.num_patches and len(selected_coords) > 0:
            # Duplicate the first patch
            selected_coords.append(selected_coords[0])
        
        # If we couldn't select any patches at all (very small image), create one centered patch
        if len(selected_coords) == 0:
            center_x, center_y = W // 2, H // 2
            size = min(W, H) // 2
            x1 = center_x - size // 2
            y1 = center_y - size // 2
            x2 = x1 + size
            y2 = y1 + size
            selected_coords.append((int(x1), int(y1), int(x2), int(y2)))
            # Duplicate this patch if needed
            while len(selected_coords) < self.num_patches:
                selected_coords.append(selected_coords[0])
        
        return selected_coords
    