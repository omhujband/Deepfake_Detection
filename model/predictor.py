"""
Predictor with support for external pre-trained models
"""

import torch
import numpy as np
import os
from .cnn_model import DeepfakeDetectorCNN
from utils.image_processor import ImageProcessor
from utils.video_processor import VideoProcessor


class DeepfakePredictor:
    """Main prediction class for deepfake detection"""
    
    def __init__(self, model_path=None, device='cuda', image_size=224, architecture='efficientnet_b0'):
        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.image_size = image_size
        self.architecture = architecture
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        
        # Try to load model
        if model_path:
            self.model_loaded = self._try_load_model(model_path)
        
        # If no model loaded, create untrained model
        if not self.model_loaded:
            print("Creating new untrained model...")
            self.model = DeepfakeDetectorCNN(
                num_classes=2, 
                pretrained=True,
                backbone=architecture
            )
            print("WARNING: Model is untrained. Results will be random.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize processors
        self.image_processor = ImageProcessor(image_size, str(self.device))
        self.video_processor = VideoProcessor(image_size, str(self.device))
    
    def _try_load_model(self, model_path):
        """Attempt to load model from path"""
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine what type of checkpoint this is
            if isinstance(checkpoint, dict):
                # Check if it's our format
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Detect architecture from weights
            architecture = self._detect_architecture(state_dict)
            print(f"Detected architecture: {architecture}")
            
            # Build model with detected architecture
            self.model = DeepfakeDetectorCNN(
                num_classes=2,
                pretrained=False,
                backbone=architecture
            )
            
            # Load weights
            self.model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will create untrained model instead.")
            return False
    
    def _detect_architecture(self, state_dict):
        """Detect architecture from state dict keys"""
        keys_str = str(list(state_dict.keys()))
        
        if 'backbone.features' in keys_str or 'efficientnet' in keys_str.lower():
            return 'efficientnet_b0'
        elif 'layer4' in keys_str:
            return 'resnet50'
        elif 'Mixed' in keys_str:
            return 'xception'
        
        return 'efficientnet_b0'
    
    def predict_image(self, image_path):
        """Predict if an image is deepfake"""
        tensor, face_found = self.image_processor.preprocess(image_path)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()
            
            prediction = 'Fake' if fake_prob > 0.5 else 'Real'
            confidence = max(fake_prob, real_prob)
        
        return {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'fake_probability': round(fake_prob * 100, 2),
            'real_probability': round(real_prob * 100, 2),
            'face_detected': face_found,
            'model_loaded': self.model_loaded
        }
    
    def predict_video(self, video_path):
        """Predict if a video is deepfake"""
        batch, metadata = self.video_processor.process_video(video_path)
        
        if batch is None:
            return {
                'prediction': 'Unknown',
                'confidence': 0,
                'error': 'No frames could be processed',
                'metadata': metadata
            }
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            
            frame_predictions = probabilities[:, 1].cpu().numpy()
            avg_fake_prob = float(np.mean(frame_predictions))
            
            fake_votes = np.sum(frame_predictions > 0.5)
            total_frames = len(frame_predictions)
            voting_ratio = fake_votes / total_frames
            
            prediction = 'Fake' if avg_fake_prob > 0.5 else 'Real'
            confidence = max(avg_fake_prob, 1 - avg_fake_prob)
        
        return {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'fake_probability': round(avg_fake_prob * 100, 2),
            'real_probability': round((1 - avg_fake_prob) * 100, 2),
            'frames_analyzed': total_frames,
            'fake_frame_ratio': round(voting_ratio * 100, 2),
            'frame_predictions': frame_predictions.tolist(),
            'metadata': metadata,
            'model_loaded': self.model_loaded
        }
    
    def get_model_info(self):
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.image_size,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'model_loaded': self.model_loaded,
            'architecture': self.architecture
        }