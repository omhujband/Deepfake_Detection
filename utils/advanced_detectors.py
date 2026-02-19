"""
Advanced Detection Methods for Modern AI-Generated Images
"""

import numpy as np
import cv2
from PIL import Image, ExifTags
from scipy import fftpack
from scipy.stats import entropy
import os


class FrequencyAnalyzer:
    """
    Detect AI-generated images using frequency domain analysis
    AI generators often leave patterns in high-frequency components
    """
    
    def __init__(self):
        self.name = "Frequency Domain Analyzer"
    
    def analyze(self, image_path):
        """
        Analyze image in frequency domain
        Returns: (is_fake_probability, confidence, details)
        """
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.5, 0, "Could not load image"
            
            # Apply FFT
            f_transform = fftpack.fft2(img)
            f_shift = fftpack.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Analyze high-frequency components
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            
            # Define high-frequency region (outer 40%)
            mask = np.ones((rows, cols), dtype=bool)
            r = int(min(rows, cols) * 0.3)
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
            mask[mask_area] = False
            
            # Calculate energy in high frequencies
            high_freq_energy = np.sum(magnitude_spectrum[mask])
            total_energy = np.sum(magnitude_spectrum)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # AI images typically have lower high-frequency content
            # Natural photos: 0.15-0.30, AI images: 0.05-0.15
            if high_freq_ratio < 0.12:
                fake_prob = 0.7  # Likely AI
                confidence = 0.6
                reason = f"Low high-freq ratio: {high_freq_ratio:.4f} (AI-like)"
            elif high_freq_ratio < 0.18:
                fake_prob = 0.5  # Uncertain
                confidence = 0.3
                reason = f"Medium high-freq ratio: {high_freq_ratio:.4f} (uncertain)"
            else:
                fake_prob = 0.3  # Likely real
                confidence = 0.6
                reason = f"High high-freq ratio: {high_freq_ratio:.4f} (natural-like)"
            
            return fake_prob, confidence, reason
            
        except Exception as e:
            return 0.5, 0, f"Frequency analysis error: {e}"


class NoiseAnalyzer:
    """
    Analyze noise patterns - AI images have unnaturally uniform noise
    """
    
    def __init__(self):
        self.name = "Noise Pattern Analyzer"
    
    def analyze(self, image_path):
        """
        Analyze noise consistency
        Returns: (is_fake_probability, confidence, details)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.5, 0, "Could not load image"
            
            # Convert to LAB color space (better for noise analysis)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply Gaussian blur and subtract to get noise
            blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
            noise = cv2.absdiff(l_channel, blurred)
            
            # Calculate noise statistics in different regions
            h, w = noise.shape
            regions = [
                noise[0:h//2, 0:w//2],      # Top-left
                noise[0:h//2, w//2:w],      # Top-right
                noise[h//2:h, 0:w//2],      # Bottom-left
                noise[h//2:h, w//2:w]       # Bottom-right
            ]
            
            # Calculate standard deviation of noise in each region
            noise_stds = [np.std(region) for region in regions]
            
            # Natural images have varying noise, AI images have uniform noise
            noise_variance = np.var(noise_stds)
            
            # Natural: variance > 5, AI: variance < 2
            if noise_variance < 2.0:
                fake_prob = 0.65
                confidence = 0.5
                reason = f"Uniform noise: {noise_variance:.2f} (AI-like)"
            elif noise_variance < 5.0:
                fake_prob = 0.5
                confidence = 0.3
                reason = f"Moderate noise variance: {noise_variance:.2f}"
            else:
                fake_prob = 0.35
                confidence = 0.5
                reason = f"Natural noise variance: {noise_variance:.2f}"
            
            return fake_prob, confidence, reason
            
        except Exception as e:
            return 0.5, 0, f"Noise analysis error: {e}"


class MetadataAnalyzer:
    """
    Check image metadata for AI generation signatures
    """
    
    def __init__(self):
        self.name = "Metadata Analyzer"
        
        # Known AI generation software signatures
        self.ai_signatures = [
            'midjourney', 'dall-e', 'dalle', 'stable diffusion', 'stablediffusion',
            'openai', 'discord', 'artificial', 'generated', 'synthetic',
            'runway', 'leonardo.ai', 'firefly', 'adobe firefly'
        ]
    
    def analyze(self, image_path):
        """
        Check EXIF and metadata for AI signatures
        Returns: (is_fake_probability, confidence, details)
        """
        try:
            img = Image.open(image_path)
            
            # Get EXIF data
            exif_data = img._getexif()
            
            if exif_data is None:
                # No EXIF data - suspicious for modern cameras but normal for screenshots
                return 0.55, 0.2, "No EXIF data (could be screenshot or AI)"
            
            # Check all EXIF fields
            all_text = ""
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                all_text += f"{tag}:{value} ".lower()
            
            # Check for AI signatures
            for signature in self.ai_signatures:
                if signature in all_text:
                    return 0.95, 0.9, f"AI signature found: '{signature}'"
            
            # Check for common AI image characteristics
            software = exif_data.get(0x0131, "").lower()  # Software tag
            
            if software:
                if any(ai in software for ai in self.ai_signatures):
                    return 0.9, 0.85, f"AI software detected: {software}"
                else:
                    return 0.3, 0.5, f"Camera software: {software}"
            
            return 0.4, 0.3, "EXIF present, no AI signatures"
            
        except Exception as e:
            return 0.5, 0, f"Metadata analysis error: {e}"


class ColorConsistencyAnalyzer:
    """
    AI images often have unnatural color distributions
    """
    
    def __init__(self):
        self.name = "Color Consistency Analyzer"
    
    def analyze(self, image_path):
        """
        Analyze color distribution consistency
        Returns: (is_fake_probability, confidence, details)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.5, 0, "Could not load image"
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalize histograms
            h_hist = h_hist.flatten() / h_hist.sum()
            s_hist = s_hist.flatten() / s_hist.sum()
            v_hist = v_hist.flatten() / v_hist.sum()
            
            # Calculate entropy (measure of randomness)
            h_entropy = entropy(h_hist + 1e-10)
            s_entropy = entropy(s_hist + 1e-10)
            v_entropy = entropy(v_hist + 1e-10)
            
            avg_entropy = (h_entropy + s_entropy + v_entropy) / 3
            
            # AI images tend to have lower entropy (more uniform colors)
            # Natural: 4.5-6.5, AI: 3.0-4.5
            if avg_entropy < 3.8:
                fake_prob = 0.65
                confidence = 0.5
                reason = f"Low color entropy: {avg_entropy:.2f} (AI-like uniform colors)"
            elif avg_entropy < 4.8:
                fake_prob = 0.5
                confidence = 0.3
                reason = f"Medium color entropy: {avg_entropy:.2f}"
            else:
                fake_prob = 0.35
                confidence = 0.5
                reason = f"High color entropy: {avg_entropy:.2f} (natural variation)"
            
            return fake_prob, confidence, reason
            
        except Exception as e:
            return 0.5, 0, f"Color analysis error: {e}"


class EnsembleDetector:
    """
    Combines multiple detection methods for better accuracy
    """
    
    def __init__(self):
        self.methods = [
            FrequencyAnalyzer(),
            NoiseAnalyzer(),
            MetadataAnalyzer(),
            ColorConsistencyAnalyzer()
        ]
    
    def analyze(self, image_path):
        """
        Run all detection methods and combine results
        Returns: dict with ensemble results
        """
        results = {
            'methods': [],
            'ensemble_fake_prob': 0,
            'ensemble_confidence': 0,
            'votes_fake': 0,
            'votes_real': 0
        }
        
        total_weighted_prob = 0
        total_weight = 0
        
        for method in self.methods:
            fake_prob, confidence, details = method.analyze(image_path)
            
            # Weight by confidence
            weight = confidence if confidence > 0 else 0.1
            total_weighted_prob += fake_prob * weight
            total_weight += weight
            
            # Count votes
            if fake_prob > 0.5:
                results['votes_fake'] += 1
            else:
                results['votes_real'] += 1
            
            results['methods'].append({
                'name': method.name,
                'fake_probability': round(fake_prob * 100, 2),
                'confidence': round(confidence * 100, 2),
                'details': details
            })
        
        # Calculate ensemble prediction
        if total_weight > 0:
            results['ensemble_fake_prob'] = round((total_weighted_prob / total_weight) * 100, 2)
            results['ensemble_confidence'] = round((total_weight / len(self.methods)) * 100, 2)
        else:
            results['ensemble_fake_prob'] = 50.0
            results['ensemble_confidence'] = 0
        
        return results