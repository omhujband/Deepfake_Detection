import torch
import torch.nn as nn
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on manipulated regions"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return attention


class DeepfakeDetectorCNN(nn.Module):
    """
    CNN for Deepfake Detection using EfficientNet-B0 backbone
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepfakeDetectorCNN, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Get the number of features from backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
        
        # Attention module
        self.attention = SpatialAttention()
        
    def forward(self, x):
        # Apply spatial attention
        att = self.attention(x)
        x = att * x + x  # Residual attention
        
        # Forward through backbone
        return self.backbone(x)
    
    def predict_proba(self, x):
        """Return probability scores"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs