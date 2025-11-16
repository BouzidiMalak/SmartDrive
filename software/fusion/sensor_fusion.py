"""
SmartDrive ADAS - Sensor Fusion Module
Combines data from multiple sensors for enhanced decision making
"""

import numpy as np
import time
from collections import deque
import json

class SensorFusion:
    def __init__(self, buffer_size=10):
        """Initialize sensor fusion system"""
        self.buffer_size = buffer_size
        
        # Data buffers for each sensor
        self.vision_buffer = deque(maxlen=buffer_size)
        self.obd_buffer = deque(maxlen=buffer_size)
        self.fatigue_buffer = deque(maxlen=buffer_size)
        
        # Fusion weights (can be adjusted based on confidence)
        self.weights = {
            'vision': 0.5,
            'obd': 0.3,
            'fatigue': 0.2
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'collision_risk': 0.7,
            'lane_departure': 0.6,
            'driver_fatigue': 0.8,
            'aggressive_driving': 0.7
        }
        
    def add_vision_data(self, vision_data):
        """Add vision system data"""
        vision_entry = {
            'timestamp': time.time(),
            'lane_departure': vision_data.get('lane_departure_warning', False),
            'collision_warning': vision_data.get('collision_warning', False),
            'objects_detected': len(vision_data.get('detected_objects', [])),
            'closest_object_distance': self._get_closest_object_distance(vision_data),
            'confidence': vision_data.get('confidence', 1.0)
        }
        self.vision_buffer.append(vision_entry)
    
    def add_obd_data(self, obd_data):
        """Add OBD system data"""
        behavior_analysis = obd_data.get('behavior_analysis', {})
        
        obd_entry = {
            'timestamp': time.time(),
            'speed': obd_data.get('speed', 0),
            'rpm': obd_data.get('rpm', 0),
            'throttle_position': obd_data.get('throttle_position', 0),
            'aggressive_acceleration': behavior_analysis.get('aggressive_acceleration', False),
            'aggressive_braking': behavior_analysis.get('aggressive_braking', False),
            'excessive_speed': behavior_analysis.get('excessive_speed', False),
            'engine_stress': behavior_analysis.get('engine_stress', False)
        }
        self.obd_buffer.append(obd_entry)
    
    def add_fatigue_data(self, fatigue_data):
        """Add driver monitoring data"""
        fatigue_entry = {
            'timestamp': time.time(),
            'fatigue_score': fatigue_data.get('fatigue_score', 0),
            'blink_detected': fatigue_data.get('blink_detected', False),
            'yawn_detected': fatigue_data.get('yawn_detected', False),
            'ear': fatigue_data.get('ear', 0.3),
            'confidence': fatigue_data.get('confidence', 1.0)
        }
        self.fatigue_buffer.append(fatigue_entry)
    
    def _get_closest_object_distance(self, vision_data):
        """Extract closest object distance from vision data"""
        objects = vision_data.get('detected_objects', [])
        if not objects:
            return float('inf')
        
        distances = [obj.get('distance', float('inf')) for obj in objects if obj.get('distance')]
        return min(distances) if distances else float('inf')
    
    def calculate_collision_risk(self):
        """Calculate overall collision risk"""
        if not self.vision_buffer or not self.obd_buffer:
            return 0.0
        
        # Vision-based risk
        latest_vision = self.vision_buffer[-1]
        vision_risk = 0.0
        
        if latest_vision['collision_warning']:
            vision_risk += 0.8
        
        closest_distance = latest_vision['closest_object_distance']
        if closest_distance != float('inf'):
            # Higher risk for closer objects
            distance_risk = max(0, 1 - (closest_distance / 20))  # Risk increases below 20m
            vision_risk += distance_risk * 0.6
        
        # OBD-based risk
        latest_obd = self.obd_buffer[-1]
        obd_risk = 0.0
        
        speed = latest_obd['speed']
        if speed > 60:  # High speed increases collision risk
            obd_risk += (speed - 60) / 100 * 0.4
        
        if latest_obd['aggressive_acceleration'] or latest_obd['aggressive_braking']:
            obd_risk += 0.3
        
        # Combine risks with weights
        total_risk = (vision_risk * self.weights['vision'] + 
                     obd_risk * self.weights['obd'])
        
        return min(total_risk, 1.0)
    
    def calculate_lane_departure_risk(self):
        """Calculate lane departure risk"""
        if not self.vision_buffer:
            return 0.0
        
        # Count recent lane departure warnings
        recent_warnings = 0
        current_time = time.time()
        
        for entry in reversed(list(self.vision_buffer)):
            if current_time - entry['timestamp'] > 5:  # Only last 5 seconds
                break
            if entry['lane_departure']:
                recent_warnings += 1
        
        # Risk increases with frequency of warnings
        risk = min(recent_warnings / len(self.vision_buffer), 1.0)
        
        return risk
    
    def calculate_driver_fatigue_risk(self):
        """Calculate driver fatigue risk"""
        if not self.fatigue_buffer:
            return 0.0
        
        # Use latest fatigue score
        latest_fatigue = self.fatigue_buffer[-1]
        fatigue_risk = latest_fatigue['fatigue_score'] / 100
        
        # Increase risk for recent yawns or abnormal blinking
        recent_yawns = sum(1 for entry in self.fatigue_buffer if entry['yawn_detected'])
        if recent_yawns > 2:
            fatigue_risk += 0.2
        
        return min(fatigue_risk, 1.0)
    
    def calculate_aggressive_driving_risk(self):
        """Calculate aggressive driving behavior risk"""
        if not self.obd_buffer:
            return 0.0
        
        # Count aggressive behaviors in recent data
        aggressive_count = 0
        total_entries = len(self.obd_buffer)
        
        for entry in self.obd_buffer:
            if (entry['aggressive_acceleration'] or 
                entry['aggressive_braking'] or 
                entry['excessive_speed'] or 
                entry['engine_stress']):
                aggressive_count += 1
        
        risk = aggressive_count / total_entries if total_entries > 0 else 0.0
        
        return risk
    
    def generate_comprehensive_assessment(self):
        """Generate comprehensive safety assessment"""
        assessment = {
            'timestamp': time.time(),
            'risks': {
                'collision': self.calculate_collision_risk(),
                'lane_departure': self.calculate_lane_departure_risk(),
                'driver_fatigue': self.calculate_driver_fatigue_risk(),
                'aggressive_driving': self.calculate_aggressive_driving_risk()
            },
            'alerts': [],
            'recommendations': [],
            'overall_safety_score': 0.0
        }
        
        # Generate alerts based on thresholds
        risks = assessment['risks']
        
        if risks['collision'] > self.alert_thresholds['collision_risk']:
            assessment['alerts'].append({
                'type': 'COLLISION_WARNING',
                'severity': 'HIGH',
                'message': 'Collision risk detected! Maintain safe distance.'
            })
        
        if risks['lane_departure'] > self.alert_thresholds['lane_departure']:
            assessment['alerts'].append({
                'type': 'LANE_DEPARTURE',
                'severity': 'MEDIUM',
                'message': 'Lane departure detected. Return to lane.'
            })
        
        if risks['driver_fatigue'] > self.alert_thresholds['driver_fatigue']:
            assessment['alerts'].append({
                'type': 'FATIGUE_WARNING',
                'severity': 'HIGH',
                'message': 'Driver fatigue detected. Take a break.'
            })
        
        if risks['aggressive_driving'] > self.alert_thresholds['aggressive_driving']:
            assessment['alerts'].append({
                'type': 'AGGRESSIVE_DRIVING',
                'severity': 'MEDIUM',
                'message': 'Aggressive driving detected. Drive more calmly.'
            })
        
        # Generate recommendations
        if risks['collision'] > 0.3:
            assessment['recommendations'].append('Increase following distance')
        
        if risks['driver_fatigue'] > 0.5:
            assessment['recommendations'].append('Consider taking a break')
        
        if risks['aggressive_driving'] > 0.4:
            assessment['recommendations'].append('Reduce acceleration and maintain steady speed')
        
        # Calculate overall safety score (lower is better)
        weighted_risk = (
            risks['collision'] * 0.4 +
            risks['lane_departure'] * 0.2 +
            risks['driver_fatigue'] * 0.3 +
            risks['aggressive_driving'] * 0.1
        )
        
        assessment['overall_safety_score'] = max(0, 100 - (weighted_risk * 100))
        
        return assessment
    
    def get_data_summary(self):
        """Get summary of all sensor data"""
        return {
            'vision_entries': len(self.vision_buffer),
            'obd_entries': len(self.obd_buffer),
            'fatigue_entries': len(self.fatigue_buffer),
            'last_update': time.time(),
            'fusion_weights': self.weights.copy()
        }
    
    def adjust_fusion_weights(self, sensor_confidence):
        """Dynamically adjust fusion weights based on sensor confidence"""
        total_confidence = sum(sensor_confidence.values())
        
        if total_confidence > 0:
            for sensor, confidence in sensor_confidence.items():
                if sensor in self.weights:
                    self.weights[sensor] = confidence / total_confidence
    
    def export_data(self, filepath):
        """Export fusion data for analysis"""
        export_data = {
            'vision_data': list(self.vision_buffer),
            'obd_data': list(self.obd_buffer),
            'fatigue_data': list(self.fatigue_buffer),
            'weights': self.weights,
            'thresholds': self.alert_thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
