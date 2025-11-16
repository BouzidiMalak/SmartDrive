"""
SmartDrive ADAS - Object Detection Module
Detects vehicles, pedestrians, and obstacles
"""

import cv2
import numpy as np
import tensorflow.lite as tflite

class ObjectDetector:
    def __init__(self, model_path='../../models/object_model.tflite'):
        """Initialize object detection with TensorFlow Lite model"""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Object classes
        self.classes = ['vehicle', 'pedestrian', 'cyclist', 'traffic_sign']
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        
    def preprocess_frame(self, frame):
        """Preprocess frame for object detection"""
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize and normalize
        resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        
        return input_data
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        input_data = self.preprocess_frame(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        return self.process_detections(boxes, classes, scores, frame.shape)
    
    def process_detections(self, boxes, classes, scores, frame_shape, threshold=0.5):
        """Process detection results"""
        height, width = frame_shape[:2]
        detections = []
        
        for i in range(len(scores)):
            if scores[i] > threshold:
                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                left = int(xmin * width)
                top = int(ymin * height)
                right = int(xmax * width)
                bottom = int(ymax * height)
                
                detection = {
                    'class': self.classes[int(classes[i])],
                    'confidence': scores[i],
                    'bbox': (left, top, right, bottom),
                    'distance': self.estimate_distance(bottom - top, self.classes[int(classes[i])])
                }
                detections.append(detection)
        
        return detections
    
    def estimate_distance(self, bbox_height, object_class):
        """Estimate distance based on bounding box height and object type"""
        # Simplified distance estimation
        if object_class == 'vehicle':
            # Average vehicle height ~1.5m
            focal_length = 500  # Calibrated value
            real_height = 1.5
        elif object_class == 'pedestrian':
            # Average person height ~1.7m
            focal_length = 500
            real_height = 1.7
        else:
            return None
            
        if bbox_height > 0:
            distance = (real_height * focal_length) / bbox_height
            return round(distance, 1)
        return None
    
    def draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        collision_warning = False
        
        for detection in detections:
            left, top, right, bottom = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            distance = detection['distance']
            
            # Choose color based on class
            class_idx = self.classes.index(class_name) if class_name in self.classes else 0
            color = self.colors[class_idx]
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            if distance:
                label += f" ({distance}m)"
                
                # Collision warning if object is too close
                if distance < 10:
                    collision_warning = True
                    color = (0, 0, 255)  # Red for warning
            
            cv2.putText(frame, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display collision warning
        if collision_warning:
            cv2.putText(frame, "COLLISION WARNING!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame, collision_warning
