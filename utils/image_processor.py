import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class FaceDetector:
    """Simple face detector using OpenCV (no extra dependencies)"""
    
    def __init__(self):
        # Load OpenCV's pre-trained face detector
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


class ImageProcessor:
    """Process images for deepfake detection"""
    
    def __init__(self, image_size=224, device='cuda'):
        self.image_size = image_size
        self.device = device
        
        # Face detector
        self.face_detector = FaceDetector()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_face(self, image):
        """Extract largest face from image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_array = np.array(image)
        faces = self.face_detector.detect(img_array)
        
        if len(faces) > 0:
            # Get largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]
            
            # Add margin
            margin = int(0.2 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.width, x + w + margin)
            y2 = min(image.height, y + h + margin)
            
            face = image.crop((x1, y1, x2, y2))
            return face, True
        
        # No face detected, return original
        return image, False
    
    def preprocess(self, image_path):
        """Full preprocessing pipeline"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Extract face
        face_img, face_found = self.extract_face(image)
        
        # Apply transforms
        tensor = self.transform(face_img)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device), face_found