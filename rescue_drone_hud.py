# Header
# Project Name: Object Detection for Rescue Operations
# Author: Saubhagya shubham
# Chosen Model: yolov8n.pt (Nano model chosen for drone processing efficiency)


# Imports
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

def process_drone_feed(source=0):
    """
    Processes video feed, runs YOLOv8 inference, and displays a Drone HUD.
    source: 0 for webcam, or a string path to a video file.
    """
    
    # ==========================================================================
    # Initialization: Loading the .pt file and Video source
    # ==========================================================================
    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt") 
    
    cap = cv2.VideoCapture(source)
    
    # Validate if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get frame dimensions to calculate the middle 20% "Target Lock" zone
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate middle 20% boundaries (40% to 60% of the screen width)
    mid_left_bound = frame_width * 0.40
    mid_right_bound = frame_width * 0.60
    
    # Variable to track time for FPS calculation
    prev_time = 0
    
    # ==========================================================================
    # Main Loop: Inference -> Filtering -> Drawing
    # ==========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or interrupted.")
            break
            
        # Inference: Resize to 640x640 internally for speed and apply confidence threshold
        # --- DOCUMENTATION: Confidence vs. IOU ---
        # Confidence Score: The model's certainty (0.0 to 1.0) that a specific object is present in the bounding box. 
        # We set conf=0.6 to filter out weak or "ghost" detections early on.
        # IOU (Intersection over Union): A metric used to measure the overlap between two bounding boxes. 
        # It is used by Non-Maximum Suppression (NMS) to delete duplicate boxes detecting the exact same object.
        # -----------------------------------------
        results = model(frame, imgsz=640, conf=0.6, verbose=False)
        
        # Extract the Results object (contains boxes, masks, etc.)
        result = results[0]
        
        if len(result.boxes) > 0:
            # Vectorization: Extracting data to NumPy arrays to avoid slow Python list iteration
            xyxy = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = result.boxes.conf.cpu().numpy() # Confidence scores
            cls_ids = result.boxes.cls.cpu().numpy() # Class IDs
            
            # Use NumPy to calculate the Center Point (X) for all boxes simultaneously
            centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
            
            # Use NumPy to calculate simulated distance (Height of box: smaller height = further away)
            box_heights = xyxy[:, 3] - xyxy[:, 1]
            distances = 1000 / (box_heights + 1e-5) # Simulated distance calculation
            
            # Use NumPy to create a boolean mask for objects inside the middle 20%
            target_locked_mask = (centers_x >= mid_left_bound) & (centers_x <= mid_right_bound)
            
            # Drawing Loop (OpenCV requires iterating to draw individual distinct shapes)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf = confs[i]
                cls_id = int(cls_ids[i])
                object_name = model.names[cls_id]
                dist = distances[i]
                cx = int(centers_x[i])
                
                # Apply Math Logic based on the vectorized boolean mask
                if target_locked_mask[i]:
                    color = (0, 0, 255) # Red (BGR format in OpenCV)
                    status_text = f"LOCKED: {object_name}"
                else:
                    color = (0, 255, 0) # Green 
                    status_text = f"SCANNING: {object_name}"
                    
                # Visual Overlay: Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Visual Overlay: Center Point
                cv2.circle(frame, (cx, int((y1+y2)/2)), 4, color, -1)
                
                # Visual Overlay: Labels (Name, Confidence, Simulated Distance)
                label = f"{status_text} | Conf: {conf:.2f} | Dist: {dist:.1f}m"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 30
        prev_time = curr_time
        
        # Heads-Up Display (HUD) Overlays
        # Draw the target zone guidelines for the middle 20%
        cv2.line(frame, (int(mid_left_bound), 0), (int(mid_left_bound), frame_height), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (int(mid_right_bound), 0), (int(mid_right_bound), frame_height), (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        cv2.putText(frame, "DRONE HUD - RESCUE OPS", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow("Drone Feed", frame)

        # Loop Control: Break gracefully on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Shutdown signal received.")
            break

    # ==========================================================================
    # Shutdown: Releasing camera and closing windows
    # ==========================================================================
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released safely.")

# Execute the function (0 uses the default webcam)
# To test on a video, change 0 to "your_drone_video.mp4"
if __name__ == "__main__":
    process_drone_feed(0)