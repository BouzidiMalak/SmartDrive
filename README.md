# SmartDrive ADAS 🚗🤖

**Advanced Driver Assistance System for Enhanced Road Safety**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

## 🎯 Project Overview

SmartDrive ADAS is a comprehensive Advanced Driver Assistance System that combines computer vision, machine learning, and vehicle diagnostics to create a safer driving experience. Built for hackathons and real-world deployment, it provides multiple safety features in an affordable, retrofittable package.

## 🌟 Key Features

### 🛣️ Lane Detection & Departure Warning
- Real-time lane line detection using deep learning
- Intelligent lane departure alerts
- Adaptive to various road conditions and markings

### 🚧 Object Detection & Collision Avoidance
- Advanced object detection (vehicles, pedestrians, cyclists)
- Distance estimation and collision risk assessment
- Real-time audio and visual warnings

### 😴 Driver Fatigue Monitoring
- Eye tracking and blink pattern analysis
- Yawn detection using facial landmarks
- Comprehensive fatigue scoring with progressive alerts

### 🌙 Night Vision Enhancement
- Automatic low-light detection
- Image enhancement for better visibility
- Adaptive processing algorithms for night driving

### 🔧 OBD-II Vehicle Integration
- Real-time vehicle data monitoring
- Aggressive driving behavior detection
- Diagnostic trouble code reading and analysis

### 🧠 Intelligent Sensor Fusion
- Multi-sensor data integration
- Comprehensive safety assessment scoring
- Smart alert prioritization and conflict resolution

## 🏗️ System Architecture

```
SmartDrive-ADAS/
│
├── hardware/                 # Hardware components and schematics
│   ├── schematics/          # Circuit diagrams and wiring
│   ├── component-list.md    # Bill of materials
│   └── mounting-designs/    # 3D models and mounting solutions
│
├── software/                # Core application code
│   ├── vision/              # Computer vision modules
│   │   ├── lane_detection.py      # Lane detection system
│   │   ├── object_detection.py    # Object detection and tracking
│   │   └── night_mode.py          # Night vision enhancement
│   ├── driver_monitoring/   # Driver state monitoring
│   │   └── fatigue_detection.py   # Fatigue and alertness monitoring
│   ├── obd/                 # Vehicle data integration
│   │   └── obd_reader.py           # OBD-II data acquisition
│   ├── fusion/              # Sensor fusion and decision making
│   │   └── sensor_fusion.py       # Multi-sensor data fusion
│   └── main.py              # Main application entry point
│
├── models/                  # Machine learning models
│   ├── lane_model.tflite    # Lane detection model
│   ├── object_model.tflite  # Object detection model
│   └── fatigue_model.tflite # Fatigue detection model
│
├── docs/                    # Documentation and presentations
│   ├── architecture.pdf     # System architecture diagrams
│   ├── slides.pptx         # Hackathon presentation
│   └── hackathon-deliverables.md
│
├── demo/                    # Demo materials
│   ├── video/
│   │   └── 1min-demo.mp4    # System demonstration video
│   └── images/              # Screenshots and demo images
│
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenCV 4.5+
- TensorFlow Lite 2.8+
- USB camera or Raspberry Pi camera module
- (Optional) OBD-II adapter for vehicle integration

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/SmartDrive-ADAS.git
cd SmartDrive-ADAS
```

2. **Install dependencies:**
```bash
pip install opencv-python tensorflow numpy scipy
pip install obd  # For OBD-II functionality (optional)
```

3. **Run the system:**
```bash
cd software
python main.py
```

### Command Line Options

```bash
python main.py --help

Options:
  --camera INT        Camera device index (default: 0)
  --enable-obd       Enable OBD-II data reading
  --no-display      Run in headless mode
```


## 🔧 Hardware Setup

### Minimum Requirements
- **Processing Unit**: Raspberry Pi 4 (4GB) or Jetson Nano
- **Camera**: USB webcam or Pi camera (1080p recommended)
- **Display**: 7" touchscreen (optional)
- **Storage**: 32GB microSD card minimum

### Optional Components
- **OBD-II Adapter**: ELM327 Bluetooth/WiFi adapter
- **GPS Module**: For location-aware features
- **Accelerometer**: For motion sensing
- **Speaker**: For audio alerts










