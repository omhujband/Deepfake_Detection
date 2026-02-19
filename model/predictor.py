"""
Unified Predictor - Auto-detects PyTorch or TensorFlow models
"""

import torch
import numpy as np
import os


class DeepfakePredictor:
    """
    Main prediction class that automatically detects and loads
    either PyTorch (.pth) or TensorFlow (.keras) models
    """
    
    def __init__(self, model_path=None, device='cuda', image_size=224):
        self.image_size = image_size
        self.device_name = device
        self.predictor = None
        self.model_type = None
        self.model_loaded = False
        
        # Auto-detect and load model
        if model_path:
            self._auto_load_model(model_path)
        else:
            # Try to find any available model
            self._find_and_load_model()
    
    def _auto_load_model(self, model_path):
        """Automatically detect model type and load"""
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
        
        # Detect by extension
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext == '.keras' or ext == '.h5':
            self._load_tensorflow_model(model_path)
        elif ext == '.pth' or ext == '.pt':
            self._load_pytorch_model(model_path)
        else:
            print(f"Unknown model format: {ext}")
    
    def _find_and_load_model(self):
        """Search for available models"""
        
        print(f"\n{'='*60}")
        print("SEARCHING FOR DEEPFAKE DETECTION MODEL")
        print(f"{'='*60}")
        
        # Check for TensorFlow models first
        tf_paths = [
            'deepfake_detector_v1.keras',              # ← ADD THIS LINE
            'deepfake_detector_final.keras',
            'model/weights/deepfake_detector_v1.keras', # ← ADD THIS LINE
            'model/weights/deepfake_detector_final.keras',
            'model/weights/deepfake_detector.keras',
            'deepfake_detector.keras'
        ]
        
        for path in tf_paths:
            if os.path.exists(path):
                print(f"✓ Found TensorFlow model: {path}")
                self._load_tensorflow_model(path)
                return
        
        # Check for PyTorch models
        pt_paths = [
            'model/weights/deepfake_detector.pth',
            'deepfake_detector.pth'
        ]
        
        for path in pt_paths:
            if os.path.exists(path):
                print(f"✓ Found PyTorch model: {path}")
                self._load_pytorch_model(path)
                return
        
        print(" No model found!")
        print("Place your model file:")
        print("  • deepfake_detector_v1.keras (TensorFlow)")
        print("  • model/weights/deepfake_detector.pth (PyTorch)")
        print(f"{'='*60}\n")
    
    def _load_tensorflow_model(self, model_path):
        """Load TensorFlow/Keras model"""
        try:
            from model.tf_predictor import TensorFlowPredictor
            
            self.predictor = TensorFlowPredictor(
                model_path=model_path,
                image_size=self.image_size,
                device=self.device_name
            )
            
            if self.predictor.model_loaded:
                self.model_type = 'TensorFlow/Keras'
                self.model_loaded = True
                print(f"✓ TensorFlow model loaded successfully!\n")
            else:
                print(f"✗ Failed to load TensorFlow model\n")
                self.predictor = None
                
        except ImportError as e:
            print(f"✗ TensorFlow not available: {e}")
            print(f"Install with: pip install tensorflow\n")
        except Exception as e:
            print(f"✗ Error loading TensorFlow model: {e}\n")
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model"""
        try:
            from model.cnn_model import DeepfakeDetectorCNN
            from utils.image_processor import ImageProcessor
            from utils.video_processor import VideoProcessor
            
            # Set device
            if self.device_name == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            
            # Load model
            self.model = DeepfakeDetectorCNN(num_classes=2, pretrained=False)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize processors
            self.image_processor = ImageProcessor(self.image_size, str(self.device))
            self.video_processor = VideoProcessor(self.image_size, str(self.device))
            
            self.model_type = 'PyTorch'
            self.model_loaded = True
            self.predictor = self  # Use self for PyTorch
            
            print(f"✓ PyTorch model loaded successfully!\n")
            
        except Exception as e:
            print(f"✗ Error loading PyTorch model: {e}\n")
    
    def predict_image(self, image_path):
        """Predict if image is deepfake"""
        
        if not self.model_loaded or self.predictor is None:
            return {
                'error': 'No model loaded',
                'prediction': 'Unknown',
                'confidence': 0,
                'model_loaded': False
            }
        
        # Delegate to appropriate predictor
        if self.model_type == 'TensorFlow/Keras':
            result = self.predictor.predict_image(image_path)
        else:
            # PyTorch prediction
            tensor, face_found = self.image_processor.preprocess(image_path)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                fake_prob = probabilities[0][1].item()
                real_prob = probabilities[0][0].item()
                
                prediction = 'Fake' if fake_prob > 0.5 else 'Real'
                confidence = max(fake_prob, real_prob)
            
            result = {
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'face_detected': face_found,
                'model_loaded': True
            }
        
        return result
    
    def predict_video(self, video_path):
        """Predict if video contains deepfakes"""
        
        if not self.model_loaded or self.predictor is None:
            return {
                'error': 'No model loaded',
                'prediction': 'Unknown',
                'confidence': 0,
                'model_loaded': False
            }
        
        # Delegate to appropriate predictor
        if self.model_type == 'TensorFlow/Keras':
            result = self.predictor.predict_video(video_path)
        else:
            # PyTorch video prediction
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
            
            result = {
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(avg_fake_prob * 100, 2),
                'real_probability': round((1 - avg_fake_prob) * 100, 2),
                'frames_analyzed': total_frames,
                'fake_frame_ratio': round(voting_ratio * 100, 2),
                'frame_predictions': frame_predictions.tolist(),
                'metadata': metadata,
                'model_loaded': True
            }
        
        return result
    
    def get_model_info(self):
        """Get model information"""
        
        if self.predictor is None or not self.model_loaded:
            return {
                'device': 'Unknown',
                'total_parameters': 0,
                'trainable_parameters': 0,
                'input_size': self.image_size,
                'gpu_available': False,
                'gpu_name': None,
                'model_loaded': False
            }
        
        if self.model_type == 'TensorFlow/Keras':
            info = self.predictor.get_model_info()
        else:
            # PyTorch model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            
            info = {
                'device': str(self.device),
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'input_size': self.image_size,
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'model_loaded': True
            }
        
        return info