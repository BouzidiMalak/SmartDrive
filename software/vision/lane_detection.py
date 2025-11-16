"""
SmartDrive ADAS - Lane Detection Module
Detects lane lines and monitors lane departure
"""

import cv2
import numpy as np
import tensorflow.lite as tflite

class LaneDetector:
    def __init__(self, model_path='../../models/lane_model.tflite'):
        """Initialize lane detection with TensorFlow Lite model"""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def preprocess_frame(self, frame):
        """Preprocess camera frame for lane detection"""
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Define region of interest (lower half of frame)
        height, width = blur.shape
        roi = blur[height//2:, :]
        
        return roi
    
    def detect_lanes(self, frame):
        """Detect lane lines using ML model"""
        processed = self.preprocess_frame(frame)
        
        # Resize for model input
        input_shape = self.input_details[0]['shape']
        resized = cv2.resize(processed, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get lane line coordinates
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return self.process_lane_output(output_data, frame.shape)
    
    def process_lane_output(self, output, frame_shape):
        """Process model output to get lane line coordinates"""
        # Implementation would depend on model architecture
        # Returns lane line coordinates and lane departure warning
        left_lane = []  # Left lane line points
        right_lane = []  # Right lane line points
        departure_warning = False
        
        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'departure_warning': departure_warning
        }
    
    def draw_lanes(self, frame, lane_data):
        """Draw detected lanes on frame"""
        if lane_data['left_lane']:
            cv2.polylines(frame, [np.array(lane_data['left_lane'])], False, (0, 255, 0), 3)
        if lane_data['right_lane']:
            cv2.polylines(frame, [np.array(lane_data['right_lane'])], False, (0, 255, 0), 3)
            
        if lane_data['departure_warning']:
            cv2.putText(frame, "LANE DEPARTURE WARNING!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
