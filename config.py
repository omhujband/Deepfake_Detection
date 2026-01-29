import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Upload settings
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    
    # Model settings
    MODEL_PATH = 'model/weights/deepfake_detector.pth'
    DEVICE = 'cuda'  # Use 'cuda' for RTX GPU, 'cpu' for CPU
    IMAGE_SIZE = 224
    CONFIDENCE_THRESHOLD = 0.5
    
    # Video processing
    FRAMES_TO_ANALYZE = 20