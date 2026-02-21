# Rescue Drone HUD: Real-Time Object Detection


## Overview
This project simulates an AI-powered Heads-Up Display (HUD) for a search-and-rescue drone. Built using Python, OpenCV, and the YOLOv8 neural network, the system processes live video feeds to detect and track critical objects (like people and vehicles) in real-time. 

To assist drone operators in high-stakes environments, the system features an "Intelligence Filtering" mechanism that categorizes detections based on their position within the drone's field of view, alerting the operator when an object is centered and locked.

---

## Key Features

* **Real-Time Inference:** Optimized to run at 30+ FPS using the lightweight `yolov8n.pt` (Nano) model.
* **Vectorized Processing:** Leverages NumPy for high-speed matrix operations on bounding box arrays, completely avoiding slow Python iteration loops.
* **Dynamic HUD Overlay:** Displays live telemetry, including an FPS counter, object class, confidence scores, and a simulated distance metric.
* **Target Acquisition Logic:** Automatically switches object status from "Scanning" to "Target Locked" based on spatial coordinates.

---

## Core Logic & Mathematics

The script applies mathematical logic to filter and categorize raw neural network outputs:

### 1. Confidence Filtering vs. IOU
* **Confidence Threshold (0.6):** The model only processes bounding boxes where the neural network is at least 60% certain the object exists, eliminating "ghost" detections.
* **Intersection over Union (IOU):** Handled natively by YOLOv8's Non-Maximum Suppression to prevent drawing multiple boxes over the exact same object.

### 2. Spatial Targeting Zones
We calculate the center of every bounding box using the coordinates $(x_1, y_1)$ and $(x_2, y_2)$. If the center point falls within the middle 20% of the screen (the targeting reticle), the HUD visually alerts the operator.

| HUD Mode | Criteria | Visual Indicator |
| :--- | :--- | :--- |
| **Scanning** | Object center is outside the middle 20% of the screen. | Green Bounding Box |
| **Target Locked** | Object center is within the middle 20% of the screen. | Red Bounding Box & Alert |

---

## Technologies Used

* **Python 3.x:** Core programming language.
* **Ultralytics (YOLOv8):** State-of-the-art object detection model.
* **OpenCV (`cv2`):** Video stream processing and HUD drawing (rectangles, text, lines).
* **NumPy:** Vectorized mathematical calculations for bounding box centers and distances.

---

## Getting Started: Setup and Usage

### Repository Structure
```text
├── rescue_drone_hud.py      # Main application and computer vision logic
├── requirements.txt         # Python package dependencies
├── README.md                # Project documentation
└── /Screenshots             # Output images demonstrating the HUD modes