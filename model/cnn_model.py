import torch
import torch.nn as nn
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
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
    Flexible CNN for Deepfake Detection
    Supports multiple backbones for compatibility with various pre-trained models
    """
    
    def __init__(self, num_classes=2, pretrained=True, backbone='efficientnet_b0'):
        super(DeepfakeDetectorCNN, self).__init__()
        
        self.backbone_name = backbone
        
        # Select backbone architecture
        if backbone == 'efficientnet_b0':
            self._build_efficientnet_b0(num_classes, pretrained)
        elif backbone == 'resnet50':
            self._build_resnet50(num_classes, pretrained)
        elif backbone == 'xception':
            self._build_xception(num_classes, pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Attention module
        self.attention = SpatialAttention()
    
    def _build_efficientnet_b0(self, num_classes, pretrained):
        """EfficientNet-B0 backbone"""
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.efficientnet_b0(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        
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
    
    def _build_resnet50(self, num_classes, pretrained):
        """ResNet-50 backbone"""
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    
    def _build_xception(self, num_classes, pretrained):
        """Xception-like backbone using Inception v3"""
        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        self.backbone = models.inception_v3(weights=weights, aux_logits=False)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Apply spatial attention
        att = self.attention(x)
        x = att * x + x
        
        # Forward through backbone
        return self.backbone(x)
    
    def predict_proba(self, x):
        """Return probability scores"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs