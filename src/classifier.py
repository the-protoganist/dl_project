"""Accident classifiers sharing one interface: update(frame, frame_idx, total_frames) -> prob | None."""
from collections import deque

import cv2
import numpy as np

from . import config


class MockClassifier:
    """MOCK: simulated probabilities (high in the middle of the video). No model, no weights."""
    name = "MOCK classifier (simulated probabilities)"
    frame_step = config.FRAME_STEP

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self):
        pass

    def update(self, frame, frame_idx=0, total_frames=0):
        pos = frame_idx / max(total_frames, 1)
        base = 0.88 if 0.45 <= pos <= 0.65 else 0.08
        return float(np.clip(base + self.rng.normal(0, 0.04), 0, 1))


class CnnLstmClassifier:
    """Real classifier: loads a checkpoint produced by training/train.py."""
    name = "CNN-LSTM (MobileNetV3-Small + LSTM)"

    def __init__(self, ckpt_path, device=None):
        import torch
        from .models import load_checkpoint
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ck = load_checkpoint(ckpt_path, self.device)
        self.clip_len, self.img_size, self.frame_step = ck["clip_len"], ck["img_size"], ck["frame_step"]
        self.buf = deque(maxlen=self.clip_len)

    def reset(self):
        self.buf.clear()

    def update(self, frame, frame_idx=0, total_frames=0):
        from .models import clip_to_tensor
        rgb = cv2.cvtColor(cv2.resize(frame, (self.img_size, self.img_size)), cv2.COLOR_BGR2RGB)
        self.buf.append(rgb)
        if len(self.buf) < self.clip_len:
            return None  # still filling the sequence window
        x = clip_to_tensor(np.stack(self.buf)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return float(self.torch.sigmoid(self.model(x))[0])
