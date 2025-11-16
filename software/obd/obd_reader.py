"""
SmartDrive ADAS - OBD-II Data Reader
Reads vehicle diagnostic data for safety analysis
"""

import time
import threading
try:
    import obd
except ImportError:
    print("OBD library not found. Install with: pip install obd")
    obd = None

class OBDReader:
    def __init__(self):
        """Initialize OBD-II connection"""
        self.connection = None
        self.data_buffer = {}
        self.reading = False
        self.read_thread = None
        
        # Commands to monitor
        self.commands = [
            'RPM',           # Engine RPM
            'SPEED',         # Vehicle speed
            'THROTTLE_POS',  # Throttle position
            'ENGINE_LOAD',   # Engine load
            'FUEL_LEVEL',    # Fuel level
            'COOLANT_TEMP',  # Engine coolant temperature
            'INTAKE_TEMP',   # Intake air temperature
        ]
        
    def connect(self, port=None):
        """Connect to OBD-II port"""
        if obd is None:
            print("OBD library not available")
            return False
            
        try:
            if port:
                self.connection = obd.OBD(port)
            else:
                self.connection = obd.OBD()  # Auto-detect port
                
            if self.connection.is_connected():
                print("OBD-II connected successfully")
                print(f"Protocol: {self.connection.protocol_name()}")
                return True
            else:
                print("Failed to connect to OBD-II")
                return False
                
        except Exception as e:
            print(f"OBD connection error: {e}")
            return False
    
    def start_reading(self):
        """Start continuous data reading"""
        if not self.connection or not self.connection.is_connected():
            print("No OBD connection available")
            return False
            
        self.reading = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("Started OBD data reading")
        return True
    
    def stop_reading(self):
        """Stop data reading"""
        self.reading = False
        if self.read_thread:
            self.read_thread.join()
        print("Stopped OBD data reading")
    
    def _read_loop(self):
        """Continuous reading loop"""
        while self.reading and self.connection.is_connected():
            try:
                # Read all supported commands
                for cmd_name in self.commands:
                    if hasattr(obd.commands, cmd_name):
                        cmd = getattr(obd.commands, cmd_name)
                        if self.connection.supports(cmd):
                            response = self.connection.query(cmd)
                            if response.value is not None:
                                self.data_buffer[cmd_name] = {
                                    'value': response.value,
                                    'unit': str(response.unit) if response.unit else '',
                                    'timestamp': time.time()
                                }
                
                time.sleep(0.5)  # Read every 500ms
                
            except Exception as e:
                print(f"OBD reading error: {e}")
                time.sleep(1)
    
    def get_latest_data(self):
        """Get latest OBD data"""
        return self.data_buffer.copy()
    
    def get_speed(self):
        """Get current vehicle speed"""
        if 'SPEED' in self.data_buffer:
            return self.data_buffer['SPEED']['value']
        return None
    
    def get_rpm(self):
        """Get current engine RPM"""
        if 'RPM' in self.data_buffer:
            return self.data_buffer['RPM']['value']
        return None
    
    def get_throttle_position(self):
        """Get throttle position percentage"""
        if 'THROTTLE_POS' in self.data_buffer:
            return self.data_buffer['THROTTLE_POS']['value']
        return None
    
    def analyze_driving_behavior(self):
        """Analyze driving behavior from OBD data"""
        analysis = {
            'aggressive_acceleration': False,
            'aggressive_braking': False,
            'excessive_speed': False,
            'engine_stress': False
        }
        
        # Get current values
        speed = self.get_speed()
        rpm = self.get_rpm()
        throttle = self.get_throttle_position()
        
        if speed is None or rpm is None or throttle is None:
            return analysis
        
        # Analyze aggressive acceleration
        if throttle > 80 and speed > 20:  # More than 80% throttle above 20 km/h
            analysis['aggressive_acceleration'] = True
        
        # Analyze engine stress
        if rpm > 4000:  # High RPM
            analysis['engine_stress'] = True
        
        # Analyze excessive speed (simplified - would need speed limit data)
        if speed > 120:  # Above 120 km/h
            analysis['excessive_speed'] = True
        
        # Analyze potential aggressive braking (sudden RPM drop)
        # This would require historical data comparison
        
        return analysis
    
    def get_diagnostic_codes(self):
        """Get diagnostic trouble codes"""
        if not self.connection or not self.connection.is_connected():
            return []
            
        try:
            # Get stored DTCs
            cmd = obd.commands.GET_DTC
            if self.connection.supports(cmd):
                response = self.connection.query(cmd)
                return response.value if response.value else []
        except Exception as e:
            print(f"Error reading DTCs: {e}")
        
        return []
    
    def clear_diagnostic_codes(self):
        """Clear diagnostic trouble codes"""
        if not self.connection or not self.connection.is_connected():
            return False
            
        try:
            cmd = obd.commands.CLEAR_DTC
            response = self.connection.query(cmd)
            return True
        except Exception as e:
            print(f"Error clearing DTCs: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from OBD-II"""
        self.stop_reading()
        if self.connection:
            self.connection.close()
            self.connection = None
        print("OBD-II disconnected")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.disconnect()
