"""
SmartDrive ADAS - Night Mode Vision Enhancement
Enhances visibility and detection in low-light conditions
"""

import cv2
import numpy as np

class NightModeProcessor:
    def __init__(self):
        """Initialize night mode processing"""
        self.adaptive_threshold = True
        self.histogram_equalization = True
        self.noise_reduction = True
        
    def detect_low_light(self, frame):
        """Detect if current lighting conditions require night mode"""
        # Convert to grayscale and calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Threshold for night mode activation (adjustable)
        night_threshold = 80
        
        return avg_brightness < night_threshold
    
    def enhance_frame(self, frame):
        """Enhance frame for better night visibility"""
        enhanced = frame.copy()
        
        # Apply histogram equalization
        if self.histogram_equalization:
            enhanced = self.apply_histogram_equalization(enhanced)
        
        # Apply adaptive threshold for edge enhancement
        if self.adaptive_threshold:
            enhanced = self.apply_adaptive_enhancement(enhanced)
        
        # Reduce noise
        if self.noise_reduction:
            enhanced = self.apply_noise_reduction(enhanced)
        
        return enhanced
    
    def apply_histogram_equalization(self, frame):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_adaptive_enhancement(self, frame):
        """Apply adaptive enhancement for edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Combine with original frame
        adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(frame, 0.8, adaptive_bgr, 0.2, 0)
        
        return enhanced
    
    def apply_noise_reduction(self, frame):
        """Apply noise reduction filter"""
        # Use bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        
        return denoised
    
    def enhance_infrared_simulation(self, frame):
        """Simulate infrared vision enhancement"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thermal-like colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        
        # Blend with original
        enhanced = cv2.addWeighted(frame, 0.6, thermal, 0.4, 0)
        
        return enhanced
    
    def process_night_frame(self, frame):
        """Main processing function for night mode"""
        # Check if night mode is needed
        is_night = self.detect_low_light(frame)
        
        if is_night:
            # Apply night mode enhancements
            enhanced = self.enhance_frame(frame)
            
            # Add night mode indicator
            cv2.putText(enhanced, "NIGHT MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            return enhanced, True
        else:
            return frame, False
