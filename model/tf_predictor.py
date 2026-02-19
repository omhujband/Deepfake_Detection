"""
TensorFlow/Keras Model Predictor for Deepfake Detection
EfficientNetB0-based model
"""

import os
import numpy as np
from PIL import Image
import cv2

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Install with: pip install tensorflow")


class FaceDetector:
    """Simple face detector using OpenCV"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect(self, image):
        """Detect faces in image, returns list of (x, y, w, h)"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        return faces


class TensorFlowPredictor:
    """
    TensorFlow/Keras Deepfake Predictor
    EfficientNetB0-based model
    """
    
    def __init__(self, model_path, image_size=224, device='cuda'):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.image_size = image_size
        self.device = device
        self.model = None
        self.model_loaded = False
        self.face_detector = FaceDetector()
        
        # Load model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"Model file not found: {model_path}")
    
    def _load_model(self, model_path):
        """Load EfficientNetB0 model with your exact architecture"""
        
        print(f"\n{'='*60}")
        print("LOADING EFFICIENTNETB0 DEEPFAKE DETECTOR")
        print(f"{'='*60}")
        print(f"Model file: {model_path}")
        
        try:
            # Import the architecture builder
            from model.efficientnet_loader import load_trained_model
            
            # Load model with exact architecture
            self.model, weights_loaded = load_trained_model(model_path)
            
            if weights_loaded:
                print("✓ Model loaded with trained weights!")
                self.model_loaded = True
            else:
                print("⚠ Model loaded but weights may not be fully trained")
                # Still mark as loaded since we have a working model
                self.model_loaded = True
            
            # Verify model works
            if self._verify_model():
                print(f"✓ Model verified and ready!")
            else:
                print(f"⚠️  Model verification failed")
                self.model_loaded = False
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def _verify_model(self):
        """Verify model works with test input"""
        try:
            print("\nVerifying model...")
            
            # Create random test image
            test_input = np.random.rand(1, self.image_size, self.image_size, 3).astype(np.float32)
            test_input = efficientnet_preprocess(test_input)
            
            # Run prediction
            prediction = self.model.predict(test_input, verbose=0)
            
            # Check output
            print(f"  Test prediction shape: {prediction.shape}")
            print(f"  Test prediction value: {prediction[0][0]:.4f}")
            
            return True
                
        except Exception as e:
            print(f"  Verification error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_face(self, image):
        """
        Extract face from image
        Returns: (PIL.Image, face_found)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_array = np.array(image)
        faces = self.face_detector.detect(img_array)
        
        if len(faces) > 0:
            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add margin
            margin = int(0.2 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.width, x + w + margin)
            y2 = min(image.height, y + h + margin)
            
            # Crop face
            face = image.crop((x1, y1, x2, y2))
            return face, True
        
        # No face found
        return image, False
    
    def _preprocess_image(self, image):
        """
        Preprocess image for EfficientNetB0 model
        
        Args:
            image: PIL Image or file path
            
        Returns:
            Preprocessed numpy array
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224)
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to array
        img_array = img_to_array(image)
        
        # Expand dimensions to create batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply EfficientNet preprocessing
        img_array = efficientnet_preprocess(img_array)
        
        return img_array
    
    def predict_image(self, image_path):
        """
        Predict if an image is deepfake
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'prediction': 'Unknown',
                'confidence': 0,
                'model_loaded': False
            }
        
        try:
            # Extract face
            face_image, face_found = self._extract_face(image_path)
            
            # Preprocess
            processed = self._preprocess_image(face_image)
            
            # Predict
            prediction = self.model.predict(processed, verbose=0)
            
            # Get probability (model outputs single sigmoid value)
            real_prob = float(prediction[0][0])
            fake_prob = 1.0 - real_prob
            
            # Determine final prediction
            # Based on sigmoid output: >0.5 = Real, <=0.5 = Fake
            if real_prob > 0.5:
                prediction_label = 'Real'
                confidence = real_prob
            else:
                prediction_label = 'Fake'
                confidence = fake_prob
            
            return {
                'prediction': prediction_label,
                'confidence': round(confidence * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'face_detected': face_found,
                'model_loaded': True
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0,
                'model_loaded': self.model_loaded
            }
    
    def predict_video(self, video_path, frames_to_analyze=20):
        """
        Predict if video contains deepfakes
        
        Args:
            video_path: Path to video file
            frames_to_analyze: Number of frames to sample
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'prediction': 'Unknown',
                'confidence': 0
            }
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {'error': 'Cannot open video', 'prediction': 'Error'}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frame indices
            if total_frames <= frames_to_analyze:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, frames_to_analyze, dtype=int)
            
            predictions = []
            face_count = 0
            
            # Process frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Extract face
                face_image, face_found = self._extract_face(pil_image)
                if face_found:
                    face_count += 1
                
                # Preprocess and predict
                processed = self._preprocess_image(face_image)
                pred = self.model.predict(processed, verbose=0)
                
                # Get probability
                real_prob = float(pred[0][0])
                predictions.append(real_prob)
            
            cap.release()
            
            # Aggregate results
            if len(predictions) == 0:
                return {
                    'error': 'No frames could be processed',
                    'prediction': 'Error'
                }
            
            avg_real_prob = np.mean(predictions)
            avg_fake_prob = 1.0 - avg_real_prob
            
            # Count fake frames
            fake_frames = sum(1 for p in predictions if p <= 0.5)
            fake_ratio = (fake_frames / len(predictions)) * 100
            
            # Overall prediction
            if avg_real_prob > 0.5:
                prediction_label = 'Real'
                confidence = avg_real_prob
            else:
                prediction_label = 'Fake'
                confidence = avg_fake_prob
            
            return {
                'prediction': prediction_label,
                'confidence': round(confidence * 100, 2),
                'real_probability': round(avg_real_prob * 100, 2),
                'fake_probability': round(avg_fake_prob * 100, 2),
                'frames_analyzed': len(predictions),
                'fake_frame_ratio': round(fake_ratio, 2),
                'frame_predictions': [round((1.0 - p) * 100, 2) for p in predictions],
                'metadata': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'duration': duration,
                    'faces_detected': face_count
                },
                'model_loaded': True
            }
            
        except Exception as e:
            print(f"Video prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence': 0
            }
    
    def get_model_info(self):
        """Get model information"""
        
        if not TF_AVAILABLE:
            return {
                'device': 'Unknown',
                'gpu_available': False,
                'gpu_name': None,
                'model_loaded': False,
                'total_parameters': 0,
                'trainable_parameters': 0,
                'input_size': self.image_size
            }
        
        # Check GPU availability
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        gpu_name = None
        
        if gpu_available:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_name = gpus[0].name
        
        info = {
            'device': 'GPU' if gpu_available else 'CPU',
            'gpu_available': gpu_available,
            'gpu_name': gpu_name,
            'model_loaded': self.model_loaded,
            'input_size': self.image_size
        }
        
        if self.model_loaded and self.model:
            try:
                params = int(self.model.count_params())
            except:
                params = 0
            
            info.update({
                'total_parameters': params,
                'trainable_parameters': params
            })
        else:
            info.update({
                'total_parameters': 0,
                'trainable_parameters': 0
            })
        
        return info

    def predict_image_enhanced(self, image_path):
        """
        Enhanced prediction using ensemble of methods
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with comprehensive analysis
        """
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'prediction': 'Unknown',
                'confidence': 0,
                'model_loaded': False
            }
        
        try:
            # Import advanced detectors
            from utils.advanced_detectors import EnsembleDetector
            
            # 1. Original model prediction
            basic_result = self.predict_image(image_path)
            
            # 2. Advanced ensemble analysis
            ensemble = EnsembleDetector()
            ensemble_result = ensemble.analyze(image_path)
            
            # 3. Combine predictions
            # Weight: 60% trained model, 40% ensemble methods
            model_weight = 0.6
            ensemble_weight = 0.4
            
            model_fake_prob = basic_result['fake_probability']
            ensemble_fake_prob = ensemble_result['ensemble_fake_prob']
            
            combined_fake_prob = (model_fake_prob * model_weight) + (ensemble_fake_prob * ensemble_weight)
            combined_real_prob = 100 - combined_fake_prob
            
            # Final prediction
            if combined_fake_prob > 50:
                final_prediction = 'Fake'
                final_confidence = combined_fake_prob
            else:
                final_prediction = 'Real'
                final_confidence = combined_real_prob
            
            return {
                'prediction': final_prediction,
                'confidence': round(final_confidence, 2),
                'fake_probability': round(combined_fake_prob, 2),
                'real_probability': round(combined_real_prob, 2),
                'face_detected': basic_result.get('face_detected', False),
                'model_loaded': True,
                
                # Detailed breakdown
                'analysis_breakdown': {
                    'cnn_model': {
                        'prediction': basic_result['prediction'],
                        'fake_prob': basic_result['fake_probability'],
                        'real_prob': basic_result['real_probability'],
                        'weight': model_weight * 100
                    },
                    'ensemble_methods': {
                        'fake_prob': ensemble_fake_prob,
                        'real_prob': round(100 - ensemble_fake_prob, 2),
                        'confidence': ensemble_result['ensemble_confidence'],
                        'weight': ensemble_weight * 100,
                        'votes_fake': ensemble_result['votes_fake'],
                        'votes_real': ensemble_result['votes_real'],
                        'methods': ensemble_result['methods']
                    }
                }
            }
            
        except Exception as e:
            print(f"Enhanced prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to basic prediction
            return self.predict_image(image_path)