"""Vehicle detection. Independent of the accident classifier (no data flows between them)."""
import cv2
import numpy as np

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck


class YoloDetector:
    name = "YOLO (Ultralytics)"

    def __init__(self, weights="yolov8n.pt", conf=0.35):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(weights), conf

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, classes=VEHICLE_CLASSES, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, self.model.names[int(b.cls)], float(b.conf)))
        return out


class MockDetector:
    """MOCK: moving-blob boxes from OpenCV background subtraction. Not a vehicle detector."""
    name = "MOCK detector (background subtraction)"

    def __init__(self, min_area=800):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
        self.min_area = min_area

    def detect(self, frame):
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            if cv2.contourArea(c) >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                out.append((x, y, x + w, y + h, "vehicle (mock)", 0.0))
        return out


def draw_detections(frame, boxes):
    img = frame.copy()
    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label} {conf:.2f}" if conf else label
        cv2.putText(img, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return img
