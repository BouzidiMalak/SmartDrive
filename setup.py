"""
SmartDrive ADAS - Setup and Installation Script
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies")
        sys.exit(1)

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera detected")
            cap.release()
        else:
            print("⚠ Warning: No camera detected. Please connect a USB camera or webcam.")
    except ImportError:
        print("⚠ Warning: OpenCV not installed. Run installation first.")

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'logs',
        'recordings',
        'exports'
    ]
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created directory: {dir_name}")

def main():
    """Main setup function"""
    print("SmartDrive ADAS Setup")
    print("=" * 20)
    
    # Check system requirements
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check hardware
    check_camera()
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nTo run SmartDrive ADAS:")
    print("  cd software")
    print("  python main.py")
    print("\nFor help:")
    print("  python main.py --help")

if __name__ == "__main__":
    main()
