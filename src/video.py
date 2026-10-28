"""Video loop: YOLO detection and accident classification run side by side on the same frames."""
import cv2

from .detector import draw_detections


def process_video(path, detector, classifier, manager, out_path=None, detect_every=2):
    """Generator yielding one dict per frame (annotated frame, probability, alert state...)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    boxes, prob, idx = [], None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % detect_every == 0:
                boxes = detector.detect(frame)
            img = draw_detections(frame, boxes)
            new_incident = None
            if idx % classifier.frame_step == 0:
                p = classifier.update(frame, idx, total)
                if p is not None:
                    prob = p
                    new_incident = manager.update(p, img, idx, idx / fps)
            if manager.alert_active:
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            label = "P(accident): warming up" if prob is None else f"P(accident): {prob:.2f}"
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if manager.alert_active else (255, 255, 255), 2)
            if writer:
                writer.write(img)
            yield {"frame": img, "idx": idx, "total": total, "prob": prob, "n_vehicles": len(boxes),
                   "streak": manager.streak, "alert_active": manager.alert_active, "new_incident": new_incident}
            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
