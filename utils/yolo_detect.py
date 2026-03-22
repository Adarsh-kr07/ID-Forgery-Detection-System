# utils/yolo_detect.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_regions(image):
    results = model(image)
    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0])
            })

    return detections