"""
Universal Model Loader
Handles various pre-trained model formats
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os


class UniversalModelLoader:
    """
    Load pre-trained deepfake detection models from various sources
    """
    
    SUPPORTED_ARCHITECTURES = [
        'efficientnet_b0',
        'efficientnet_b1', 
        'efficientnet_b4',
        'resnet50',
        'resnet101',
        'xception',
        'inception_v3'
    ]
    
    def __init__(self, device='cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
    
    def load_model(self, model_path, architecture='auto'):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to .pth file
            architecture: Model architecture or 'auto' to detect
            
        Returns:
            Loaded model ready for inference
        """
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Detect architecture if auto
        if architecture == 'auto':
            architecture = self._detect_architecture(checkpoint)
            print(f"Detected architecture: {architecture}")
        
        # Build model
        model = self._build_model(architecture, checkpoint)
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def _detect_architecture(self, checkpoint):
        """Auto-detect model architecture from checkpoint"""
        
        # Get state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Check layer names to detect architecture
        keys = list(state_dict.keys())
        first_key = keys[0] if keys else ''
        
        if 'efficientnet' in first_key.lower() or '_fc' in str(keys):
            if any('b4' in k.lower() for k in keys):
                return 'efficientnet_b4'
            elif any('b1' in k.lower() for k in keys):
                return 'efficientnet_b1'
            return 'efficientnet_b0'
        
        elif 'layer1' in str(keys) and 'layer4' in str(keys):
            # ResNet architecture
            if any('2048' in str(state_dict[k].shape) for k in keys if 'fc' in k):
                return 'resnet50'
            return 'resnet50'
        
        elif 'Mixed' in str(keys) or 'inception' in first_key.lower():
            return 'inception_v3'
        
        # Default to EfficientNet-B0
        print("Warning: Could not detect architecture, defaulting to efficientnet_b0")
        return 'efficientnet_b0'
    
    def _build_model(self, architecture, checkpoint):
        """Build model with correct architecture and load weights"""
        
        # Get state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Detect number of classes
        num_classes = self._detect_num_classes(state_dict)
        print(f"Detected {num_classes} output classes")
        
        # Build appropriate model
        if architecture == 'efficientnet_b0':
            model = self._build_efficientnet(state_dict, 'b0', num_classes)
        elif architecture == 'efficientnet_b1':
            model = self._build_efficientnet(state_dict, 'b1', num_classes)
        elif architecture == 'efficientnet_b4':
            model = self._build_efficientnet(state_dict, 'b4', num_classes)
        elif architecture == 'resnet50':
            model = self._build_resnet(state_dict, 50, num_classes)
        elif architecture == 'resnet101':
            model = self._build_resnet(state_dict, 101, num_classes)
        elif architecture == 'inception_v3':
            model = self._build_inception(state_dict, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        return model
    
    def _detect_num_classes(self, state_dict):
        """Detect number of output classes"""
        for key in reversed(list(state_dict.keys())):
            if 'weight' in key and ('fc' in key or 'classifier' in key):
                shape = state_dict[key].shape
                if len(shape) == 2:
                    return shape[0]
        return 2  # Default to binary classification
    
    def _build_efficientnet(self, state_dict, variant, num_classes):
        """Build EfficientNet model"""
        
        if variant == 'b0':
            model = models.efficientnet_b0(weights=None)
            num_features = 1280
        elif variant == 'b1':
            model = models.efficientnet_b1(weights=None)
            num_features = 1280
        elif variant == 'b4':
            model = models.efficientnet_b4(weights=None)
            num_features = 1792
        
        # Try to match classifier structure
        model.classifier = self._build_classifier(state_dict, num_features, num_classes)
        
        # Load weights
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Partial weight loading: {e}")
        
        return model
    
    def _build_resnet(self, state_dict, depth, num_classes):
        """Build ResNet model"""
        
        if depth == 50:
            model = models.resnet50(weights=None)
        elif depth == 101:
            model = models.resnet101(weights=None)
        
        num_features = model.fc.in_features
        model.fc = self._build_classifier(state_dict, num_features, num_classes)
        
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Partial weight loading: {e}")
        
        return model
    
    def _build_inception(self, state_dict, num_classes):
        """Build Inception model"""
        
        model = models.inception_v3(weights=None, aux_logits=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Partial weight loading: {e}")
        
        return model
    
    def _build_classifier(self, state_dict, num_features, num_classes):
        """Build classifier head matching the checkpoint"""
        
        # Try to detect classifier structure
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k]
        
        # Count linear layers
        linear_count = sum(1 for k in classifier_keys if 'weight' in k and 
                         len(state_dict[k].shape) == 2)
        
        if linear_count >= 3:
            # Complex classifier (like our custom one)
            return nn.Sequential(
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
        else:
            # Simple classifier
            return nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )


# Wrapper class for easy use
class PretrainedDeepfakeDetector(nn.Module):
    """
    Easy-to-use wrapper for pre-trained deepfake detection models
    """
    
    def __init__(self, model_path, architecture='auto', device='cuda'):
        super(PretrainedDeepfakeDetector, self).__init__()
        
        loader = UniversalModelLoader(device)
        self.model = loader.load_model(model_path, architecture)
        self.device = loader.device
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        """Get prediction (Real/Fake)"""
        self.eval()
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def predict_proba(self, x):
        """Get probability scores"""
        self.eval()
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
        return probs