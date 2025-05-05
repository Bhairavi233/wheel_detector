import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
import pandas as pd
from datetime import datetime

# Load YOLOv5 models
@st.cache_resource
def load_yolo5_models():
    vehicle_model = torch.hub.load('yolov5', 'custom',
                                    path='C:/Users/Hp/Downloads/projects/Vehicle-Detection-main/Vehicle-Detection-main/runs/train/exp12/weights/best.pt', source='local')
    wheel_model = torch.hub.load('yolov5', 'custom',
                                  path='C:/Users/Hp/Downloads/projects/wheel-detector-main/wheel-detector-main/wheel_detector.pt', source='local')
    return vehicle_model, wheel_model

vehicle_model, wheel_model = load_yolo5_models()

# UI
st.title("Multi-YOLO Vehicle Detection System")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
save_csv = st.sidebar.checkbox("Save Detection Results to CSV")
uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4"])

# Detection function
def process_all_models(image, conf=0.25):
    vehicle_model.conf = conf
    wheel_model.conf = conf

    results_vehicle = vehicle_model(image)
    results_wheel = wheel_model(image)

    vehicle_img = np.squeeze(results_vehicle.render())
    wheel_img = np.squeeze(results_wheel.render())

    return results_vehicle, results_wheel, vehicle_img, wheel_img

# CSV logging
def save_detections_to_csv(results, label="vehicle"):
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        detections.append({
            "label": label,
            "x1": int(xyxy[0]),
            "y1": int(xyxy[1]),
            "x2": int(xyxy[2]),
            "y2": int(xyxy[3]),
            "confidence": float(conf),
            "class": int(cls)
        })

    if detections:
        df = pd.DataFrame(detections)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections_{label}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        st.success(f"{label.capitalize()} results saved to {filename}")

# Main logic
if uploaded_file:
    file_type = uploaded_file.type

    if "image" in file_type:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        image_np = np.array(image)

        results_vehicle, results_wheel, vehicle_img, wheel_img = process_all_models(image_np, conf=conf_threshold)

        st.subheader("Vehicle Classification (YOLOv5)")
        st.image(vehicle_img, channels="BGR", use_column_width=True)

        st.subheader("Wheel Detection (YOLOv5)")
        st.image(wheel_img, channels="BGR", use_column_width=True)

        if save_csv:
            save_detections_to_csv(results_vehicle, "vehicle")
            save_detections_to_csv(results_wheel, "wheel")

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        all_vehicle_detections = []
        all_wheel_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results_vehicle, results_wheel, _, wheel_img = process_all_models(frame, conf=conf_threshold)
            stframe.image(wheel_img, channels="BGR")

            if save_csv:
                for *xyxy, conf, cls in results_vehicle.xyxy[0]:
                    all_vehicle_detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)])

                for *xyxy, conf, cls in results_wheel.xyxy[0]:
                    all_wheel_detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)])

        cap.release()
        st.success("Video processing complete.")

        # Save at the end
        if save_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pd.DataFrame(all_vehicle_detections, columns=["x1", "y1", "x2", "y2", "conf", "class"]).to_csv(f"vehicle_detections_{timestamp}.csv", index=False)
            pd.DataFrame(all_wheel_detections, columns=["x1", "y1", "x2", "y2", "conf", "class"]).to_csv(f"wheel_detections_{timestamp}.csv", index=False)
            st.success("CSV files saved.")

