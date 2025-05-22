# AI-Based Video Surveillance System
# Features: Real-time Object Detection, Tracking, Unusual Activity Alerts, Optional License Plate Recognition

# 1. Install dependencies before running:
# pip install ultralytics opencv-python deep_sort_realtime easyocr numpy

import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

# Initialize YOLOv8 model
detector = YOLO("yolov8n.pt")  # or "yolov8s.pt" for better accuracy

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# Optional: Initialize OCR for license plate recognition
ocr_reader = easyocr.Reader(['en'])

# Load camera
cap = cv2.VideoCapture(0)  # or use an IP camera URL

# Store dwell times and IDs
dwell_start = {}
UNUSUAL_TIME = 10  # seconds to trigger loitering alert

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = detector(frame)[0]
    detections = []

    for result in results.boxes:
        cls = int(result.cls[0])
        conf = float(result.conf[0])
        if conf < 0.3:
            continue

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = detector.names[cls]

        if label == "person":
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # DeepSort expects (x, y, w, h), confidence, class
    tracks = tracker.update_tracks(detections, frame=frame)

    current_time = time.time()

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, l + w, t + h])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Loitering detection
        if track_id not in dwell_start:
            dwell_start[track_id] = current_time
        else:
            duration = current_time - dwell_start[track_id]
            if duration > UNUSUAL_TIME:
                cv2.putText(frame, "Loitering Detected!", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("AI Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
