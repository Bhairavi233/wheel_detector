import cv2
import torch
import numpy as np
import csv
import math
from sort import Sort
from collections import defaultdict
import time
start = time.time()
import os

video_path = 'C:/Users/Hp/Downloads/video10.mp4'
cap = cv2.VideoCapture(video_path)
video_filename = os.path.basename(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0

# Load models
vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Hp/Downloads/projects/Vehicle-Detection-main/Vehicle-Detection-main/runs/train/exp12/weights/best.pt')
wheel_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Hp/Downloads/projects/wheel-detector-main/wheel-detector-main/wheel_detector.pt')

vehicle_model.conf = 0.3
wheel_model.conf = 0.3

# Class names
VEHICLE_CLASSES = ['Car', 'Truck']
WHEEL_CLASS = 'wheel'

# Tracker
tracker = Sort()

# CSV logging
csv_data = []

# ROI polygon for center toll lane (adjust as needed)
ROI_POLYGON = np.array([
    [171, 471], 
    [187, 2], 
    [703, 2], 
    [703, 286],
    [1066, 379]
  # [163, 454],
  #  [169, 20]
])

# Helper functions
def is_inside(wheel_box, vehicle_box):
    x1_w, y1_w, x2_w, y2_w = wheel_box
    x1_v, y1_v, x2_v, y2_v = vehicle_box
    return x1_v <= x1_w <= x2_w <= x2_v and y1_v <= y1_w <= y2_w <= y2_v

def is_inside_lane(x_center, y_center, polygon):
    return cv2.pointPolygonTest(polygon, (x_center, y_center), False) >= 0

# Video input
#cap = cv2.VideoCapture('C:/Users/Hp/Downloads/vehicle1.mp4')
#frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # Draw ROI polygon
    cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

    # Vehicle detection
    vehicle_results = vehicle_model(frame, size=320)#size=320,size=640
    vehicle_detections = vehicle_results.xyxy[0].cpu().numpy()

    vehicles_for_tracking = []
    vehicle_boxes_for_assoc = []

    for *box, conf, cls_id in vehicle_detections:
        class_name = vehicle_model.names[int(cls_id)]
        if class_name in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            if is_inside_lane(x_center, y_center, ROI_POLYGON):
                vehicles_for_tracking.append([x1, y1, x2, y2, conf])
                vehicle_boxes_for_assoc.append([x1, y1, x2, y2])

    # Wheel detection
    wheel_results = wheel_model(frame, size=320)#size=320,size=640
    wheel_detections = wheel_results.xyxy[0].cpu().numpy()
    wheel_boxes = []
    tracked_vehicles = []

    for *box, conf, cls_id in wheel_detections:
        class_name = wheel_model.names[int(cls_id)]
        if class_name == WHEEL_CLASS:
            x1, y1, x2, y2 = map(int, box)
            wheel_boxes.append([x1, y1, x2, y2])

    # Run SORT tracker
    if len(vehicles_for_tracking) > 0:
        detections_np = np.array(vehicles_for_tracking, dtype=np.float32)
        tracked_vehicles = tracker.update(detections_np)
    else:
        tracked_vehicles = np.empty((0, 5))

    # Associate wheels with tracked vehicles
    for track in tracked_vehicles:
        x1, y1, x2, y2, track_id = map(int, track)
        box = [x1, y1, x2, y2]
        matched_wheels = [w for w in wheel_boxes if is_inside(w, box)]

        # Draw vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id} Wheels:{len(matched_wheels)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Save CSV row
        timestamp = frame_id / fps
        csv_data.append([video_filename, timestamp, frame_id, track_id, len(matched_wheels), x1, y1, x2, y2])
        
    # Optional: draw wheel boxes
    for x1, y1, x2, y2 in wheel_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Display frame
    cv2.imshow("Vehicle + Wheel Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save final deduplicated summary
final_vehicle_data = {}
for video_name, timestamp, frame_id, track_id, wheel_count, x1, y1, x2, y2 in csv_data:
    final_vehicle_data[track_id] = [video_name, timestamp, frame_id, track_id, wheel_count, x1, y1, x2, y2]

with open('vehicle_wheel_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['video_name', 'timestamp_sec', 'frame_id', 'vehicle_id', 'wheel_count', 'x1', 'y1', 'x2', 'y2'])
    writer.writerows(csv_data)
end = time.time()
duration = end - start    