import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GlobalUncertaintyEstimator(nn.Module):
    """
    Global model that performs initial classification and uncertainty estimation
    """
    def __init__(self, num_classes, backbone='resnet34', pretrained=True):
        super(GlobalUncertaintyEstimator, self).__init__()
        
        # Load backbone with pretrained weights if specified
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer to accept grayscale images
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Uncertainty estimation head using evidential learning
        self.evidence_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4 * num_classes, kernel_size=1),  # 4 parameters per class: alpha, beta, gamma, v
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Global classification
        logits = self.classifier(features)  # [B, num_classes]
        
        # Evidence parameters for uncertainty estimation
        evidence_params = self.evidence_head(features)
        
        # Split evidence parameters
        batch_size, channels, h, w = evidence_params.shape
        num_classes = logits.size(1)
        
        # Reshape to get the 4 EDL parameters for each class
        evidence_params = evidence_params.view(batch_size, 4, num_classes, h, w)
        
        # Get individual EDL parameters (alpha, beta, etc.)
        alpha = F.softplus(evidence_params[:, 0]) + 1.0  # alpha > 1
        beta = F.softplus(evidence_params[:, 1])  # beta > 0
        
        # Compute uncertainty map (higher value = more uncertain)
        # Calculate uncertainty for each class
        uncertainty_per_class = 1.0 / alpha + beta / (alpha * (alpha + 1.0))  # [B, num_classes, H, W]
        
        # Average across classes to get a single uncertainty map
        uncertainty_map = uncertainty_per_class.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Normalize to [0, 1] for easier interpretation
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-6)
        
        return logits, uncertainty_map, (alpha, beta)
    