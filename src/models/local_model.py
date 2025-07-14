import torch
import torch.nn as nn

class LocalRefinementNetwork(nn.Module):
    """
    Network for analyzing patches in detail
    """
    def __init__(self, num_classes, patch_size=64):
        super(LocalRefinementNetwork, self).__init__()
        
        # Feature extractor for patches
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalize confidence to [0, 1]
        )
    
    def forward(self, patches):
        """
        Args:
            patches: Tensor of shape [B, K, C, H, W] where K is the number of patches
        
        Returns:
            patch_logits: Tensor of shape [B, K, num_classes]
            patch_confidences: Tensor of shape [B, K, 1]
        """
        batch_size, num_patches, channels, height, width = patches.shape
        
        # Reshape to process all patches at once
        patches_flat = patches.view(-1, channels, height, width)
        
        # Extract features
        features = self.patch_encoder(patches_flat)  # [B*K, 256, 1, 1]
        
        # Get logits and confidences
        logits = self.classifier(features)  # [B*K, num_classes]
        confidences = self.confidence_head(features)  # [B*K, 1]
        
        # Reshape back to batch format
        patch_logits = logits.view(batch_size, num_patches, -1)  # [B, K, num_classes]
        patch_confidences = confidences.view(batch_size, num_patches, -1)  # [B, K, 1]
        
        return patch_logits, patch_confidences
    