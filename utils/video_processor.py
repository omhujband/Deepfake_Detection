import cv2
import numpy as np
from PIL import Image
import torch
from .image_processor import ImageProcessor


class VideoProcessor:
    """Process videos for deepfake detection"""
    
    def __init__(self, image_size=224, device='cuda', frames_to_analyze=20):
        self.image_size = image_size
        self.device = device
        self.frames_to_analyze = frames_to_analyze
        self.image_processor = ImageProcessor(image_size, device)
    
    def extract_frames(self, video_path):
        """Extract frames uniformly from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame indices to sample
        if total_frames <= self.frames_to_analyze:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(
                0, total_frames - 1, self.frames_to_analyze, dtype=int
            )
        
        frames = []
        face_detected_count = 0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                face_img, face_found = self.image_processor.extract_face(pil_image)
                if face_found:
                    face_detected_count += 1
                
                frames.append(face_img)
        
        cap.release()
        
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'frames_analyzed': len(frames),
            'faces_detected': face_detected_count
        }
        
        return frames, metadata
    
    def preprocess_frames(self, frames):
        """Preprocess extracted frames for model input"""
        tensors = []
        
        for frame in frames:
            tensor = self.image_processor.transform(frame)
            tensors.append(tensor)
        
        if tensors:
            batch = torch.stack(tensors).to(self.device)
            return batch
        
        return None
    
    def process_video(self, video_path):
        """Full video processing pipeline"""
        frames, metadata = self.extract_frames(video_path)
        
        if not frames:
            return None, metadata
        
        batch = self.preprocess_frames(frames)
        return batch, metadata