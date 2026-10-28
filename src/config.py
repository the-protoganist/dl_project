"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "incidents.db"
EVIDENCE_DIR = ROOT / "data" / "evidence"
OUTPUT_DIR = ROOT / "data" / "output"
CHECKPOINT_PATH = ROOT / "models" / "accident_cnn_lstm.pt"
YOLO_WEIGHTS = "yolov8n.pt"  # downloaded automatically by Ultralytics on first real run

CLIP_LEN = 16      # frames per classifier sequence
IMG_SIZE = 224     # classifier input resolution
FRAME_STEP = 5     # classifier consumes every Nth video frame
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
