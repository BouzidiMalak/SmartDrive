"""
SmartDrive ADAS - Driver Fatigue Detection
Monitors driver alertness using eye tracking and behavior analysis
"""

import cv2
import numpy as np
import time
import tensorflow.lite as tflite
from scipy.spatial import distance as dist

class FatigueDetector:
    def __init__(self, model_path='../../models/fatigue_model.tflite'):
        """Initialize fatigue detection system"""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Eye aspect ratio threshold and frame counters
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.eye_counter = 0
        self.total_blinks = 0
        
        # Yawning detection
        self.MAR_THRESHOLD = 0.6
        self.yawn_counter = 0
        
        # Timing for fatigue analysis
        self.start_time = time.time()
        self.blink_timestamps = []
        
    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate eye aspect ratio for blink detection"""
        # Compute euclidean distances between vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate mouth aspect ratio for yawn detection"""
        # Vertical distances
        A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[10])
        B = dist.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        
        # Horizontal distance
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        
        # Calculate mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar
    
    def detect_face_landmarks(self, frame):
        """Detect facial landmarks using ML model"""
        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get landmarks
        landmarks = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Convert normalized coordinates to pixel coordinates
        height, width = frame.shape[:2]
        landmarks[:, 0] *= width
        landmarks[:, 1] *= height
        
        return landmarks.astype(int)
    
    def analyze_drowsiness(self, frame):
        """Analyze driver drowsiness state"""
        try:
            # Detect facial landmarks
            landmarks = self.detect_face_landmarks(frame)
            
            # Extract eye and mouth regions (assuming specific landmark indices)
            left_eye = landmarks[36:42]   # Left eye landmarks
            right_eye = landmarks[42:48]  # Right eye landmarks
            mouth = landmarks[48:68]      # Mouth landmarks
            
            # Calculate ratios
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            mar = self.mouth_aspect_ratio(mouth)
            
            # Blink detection
            blink_detected = False
            if ear < self.EAR_THRESHOLD:
                self.eye_counter += 1
            else:
                if self.eye_counter >= self.EAR_CONSEC_FRAMES:
                    self.total_blinks += 1
                    self.blink_timestamps.append(time.time())
                    blink_detected = True
                self.eye_counter = 0
            
            # Yawn detection
            yawn_detected = False
            if mar > self.MAR_THRESHOLD:
                self.yawn_counter += 1
                if self.yawn_counter >= 10:  # Sustained yawn
                    yawn_detected = True
            else:
                self.yawn_counter = 0
            
            # Calculate fatigue score
            fatigue_score = self.calculate_fatigue_score()
            
            return {
                'ear': ear,
                'mar': mar,
                'blink_detected': blink_detected,
                'yawn_detected': yawn_detected,
                'fatigue_score': fatigue_score,
                'landmarks': landmarks
            }
            
        except Exception as e:
            return None
    
    def calculate_fatigue_score(self):
        """Calculate overall fatigue score based on multiple factors"""
        current_time = time.time()
        time_elapsed = current_time - self.start_time
        
        # Calculate blink rate (blinks per minute)
        recent_blinks = [t for t in self.blink_timestamps if current_time - t <= 60]
        blink_rate = len(recent_blinks)
        
        # Normal blink rate is 15-20 per minute
        # Low blink rate (<10) or very high rate (>30) indicates fatigue
        blink_score = 0
        if blink_rate < 10:
            blink_score = (10 - blink_rate) / 10 * 50  # Up to 50 points
        elif blink_rate > 30:
            blink_score = (blink_rate - 30) / 10 * 30  # Up to 30 points
        
        # Yawn frequency
        yawn_score = self.yawn_counter * 10  # Each yawn adds 10 points
        
        # Time factor (fatigue increases over time)
        time_score = min(time_elapsed / 3600 * 20, 40)  # Up to 40 points after 2 hours
        
        total_score = min(blink_score + yawn_score + time_score, 100)
        
        return total_score
    
    def draw_fatigue_analysis(self, frame, analysis):
        """Draw fatigue analysis results on frame"""
        if analysis is None:
            cv2.putText(frame, "Face not detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Draw landmarks
        for (x, y) in analysis['landmarks']:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Display metrics
        cv2.putText(frame, f"EAR: {analysis['ear']:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {analysis['mar']:.3f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Fatigue score and alerts
        fatigue_score = analysis['fatigue_score']
        color = (0, 255, 0)  # Green
        alert_text = "ALERT"
        
        if fatigue_score > 70:
            color = (0, 0, 255)  # Red
            alert_text = "CRITICAL FATIGUE!"
        elif fatigue_score > 40:
            color = (0, 165, 255)  # Orange
            alert_text = "FATIGUE WARNING"
        
        cv2.putText(frame, f"Fatigue Score: {fatigue_score:.1f}%", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if fatigue_score > 40:
            cv2.putText(frame, alert_text, (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Specific alerts
        if analysis['blink_detected']:
            cv2.putText(frame, "BLINK", (200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if analysis['yawn_detected']:
            cv2.putText(frame, "YAWN DETECTED", (200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
