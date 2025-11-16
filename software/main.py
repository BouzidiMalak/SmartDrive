"""
SmartDrive ADAS - Main Application
Coordinates all ADAS subsystems for comprehensive driver assistance
"""

import cv2
import time
import threading
import argparse
from datetime import datetime

# Import ADAS modules
from vision.lane_detection import LaneDetector
from vision.object_detection import ObjectDetector
from vision.night_mode import NightModeProcessor
from driver_monitoring.fatigue_detection import FatigueDetector
from obd.obd_reader import OBDReader
from fusion.sensor_fusion import SensorFusion

class SmartDriveADAS:
    def __init__(self, camera_source=0, enable_obd=False):
        """Initialize SmartDrive ADAS system"""
        print("Initializing SmartDrive ADAS...")
        
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_source)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Initialize vision modules
        self.lane_detector = LaneDetector()
        self.object_detector = ObjectDetector()
        self.night_processor = NightModeProcessor()
        
        # Initialize driver monitoring
        self.fatigue_detector = FatigueDetector()
        
        # Initialize OBD reader (optional)
        self.obd_reader = None
        if enable_obd:
            self.obd_reader = OBDReader()
            if self.obd_reader.connect():
                self.obd_reader.start_reading()
        
        # Initialize sensor fusion
        self.sensor_fusion = SensorFusion()
        
        # System state
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Display settings
        self.display_debug = True
        self.display_scale = 0.8
        
        print("SmartDrive ADAS initialized successfully!")
    
    def process_frame(self, frame):
        """Process a single frame through all ADAS systems"""
        processed_frame = frame.copy()
        
        # Apply night mode enhancement if needed
        enhanced_frame, night_mode_active = self.night_processor.process_night_frame(frame)
        if night_mode_active:
            processed_frame = enhanced_frame
        
        # Lane detection
        lane_data = self.lane_detector.detect_lanes(processed_frame)
        processed_frame = self.lane_detector.draw_lanes(processed_frame, lane_data)
        
        # Object detection
        detected_objects = self.object_detector.detect_objects(processed_frame)
        processed_frame, collision_warning = self.object_detector.draw_detections(
            processed_frame, detected_objects
        )
        
        # Driver fatigue monitoring
        fatigue_analysis = self.fatigue_detector.analyze_drowsiness(frame)
        if fatigue_analysis:
            processed_frame = self.fatigue_detector.draw_fatigue_analysis(
                processed_frame, fatigue_analysis
            )
        
        # Prepare vision data for sensor fusion
        vision_data = {
            'lane_departure_warning': lane_data.get('departure_warning', False),
            'collision_warning': collision_warning,
            'detected_objects': detected_objects,
            'night_mode_active': night_mode_active,
            'confidence': 0.9  # Vision system confidence
        }
        
        # Add data to sensor fusion
        self.sensor_fusion.add_vision_data(vision_data)
        
        if fatigue_analysis:
            self.sensor_fusion.add_fatigue_data(fatigue_analysis)
        
        # Add OBD data if available
        if self.obd_reader:
            obd_data = self.obd_reader.get_latest_data()
            behavior_analysis = self.obd_reader.analyze_driving_behavior()
            obd_data['behavior_analysis'] = behavior_analysis
            self.sensor_fusion.add_obd_data(obd_data)
        
        return processed_frame
    
    def draw_system_info(self, frame):
        """Draw system information overlay"""
        height, width = frame.shape[:2]
        
        # System status box
        cv2.rectangle(frame, (10, height - 150), (300, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, height - 150), (300, height - 10), (255, 255, 255), 2)
        
        # System info
        info_y = height - 130
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        info_y += 25
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        info_y += 25
        obd_status = "Connected" if self.obd_reader and self.obd_reader.connection else "Disabled"
        cv2.putText(frame, f"OBD: {obd_status}", (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Safety assessment
        assessment = self.sensor_fusion.generate_comprehensive_assessment()
        safety_score = assessment['overall_safety_score']
        
        # Safety score display
        score_color = (0, 255, 0) if safety_score > 80 else (0, 255, 255) if safety_score > 60 else (0, 0, 255)
        cv2.putText(frame, f"Safety Score: {safety_score:.0f}%", (width - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
        
        # Active alerts
        alert_y = 70
        for alert in assessment['alerts']:
            alert_color = (0, 0, 255) if alert['severity'] == 'HIGH' else (0, 165, 255)
            cv2.putText(frame, alert['message'], (width - 400, alert_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
            alert_y += 25
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        self.frame_count += 1
    
    def run(self):
        """Main execution loop"""
        print("Starting SmartDrive ADAS...")
        print("Press 'q' to quit, 'd' to toggle debug info, 's' to save assessment")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add system information overlay
                if self.display_debug:
                    processed_frame = self.draw_system_info(processed_frame)
                
                # Update FPS
                self.update_fps()
                
                # Resize for display
                if self.display_scale != 1.0:
                    height, width = processed_frame.shape[:2]
                    new_width = int(width * self.display_scale)
                    new_height = int(height * self.display_scale)
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))
                
                # Display frame
                cv2.imshow('SmartDrive ADAS', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.display_debug = not self.display_debug
                    print(f"Debug display: {'ON' if self.display_debug else 'OFF'}")
                elif key == ord('s'):
                    self.save_assessment()
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cleanup()
    
    def save_assessment(self):
        """Save current safety assessment to file"""
        assessment = self.sensor_fusion.generate_comprehensive_assessment()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"safety_assessment_{timestamp}.json"
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(assessment, f, indent=2)
            print(f"Safety assessment saved to {filename}")
        except Exception as e:
            print(f"Failed to save assessment: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        print("Shutting down SmartDrive ADAS...")
        
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if self.obd_reader:
            self.obd_reader.disconnect()
        
        cv2.destroyAllWindows()
        print("Cleanup complete.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SmartDrive ADAS - Advanced Driver Assistance System')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument('--enable-obd', action='store_true', help='Enable OBD-II data reading')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless mode)')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run ADAS
        adas = SmartDriveADAS(camera_source=args.camera, enable_obd=args.enable_obd)
        
        if args.no_display:
            print("Running in headless mode - use Ctrl+C to stop")
            try:
                while True:
                    ret, frame = adas.camera.read()
                    if ret:
                        adas.process_frame(frame)
                    time.sleep(0.033)  # ~30 FPS
            except KeyboardInterrupt:
                pass
        else:
            adas.run()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
