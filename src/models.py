"""CNN-LSTM accident classifier: frozen MobileNetV3-Small features + trainable LSTM."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from . import config

_MEAN = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)


def clip_to_tensor(clip):
    """(T, H, W, 3) uint8 RGB array -> normalized float tensor (T, 3, H, W)."""
    x = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float() / 255.0
    return (x - _MEAN) / _STD


class CnnLstm(nn.Module):
    def __init__(self, pretrained=True, hidden=128, layers=1):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # 576-channel output
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.features.parameters():
            p.requires_grad = False  # frozen CNN
        self.hidden, self.layers = hidden, layers
        self.lstm = nn.LSTM(576, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()  # keep frozen BatchNorm statistics fixed
        return self

    def forward(self, x):  # x: (B, T, 3, H, W) -> logits (B,)
        b, t = x.shape[:2]
        with torch.no_grad():
            f = self.pool(self.features(x.flatten(0, 1))).flatten(1).view(b, t, -1)
        out, _ = self.lstm(f)
        return self.head(out[:, -1]).squeeze(1)


def save_checkpoint(model, path, metrics=None):
    path = path if hasattr(path, "parent") else __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "hidden": model.hidden, "layers": model.layers,
        "clip_len": config.CLIP_LEN, "img_size": config.IMG_SIZE,
        "frame_step": config.FRAME_STEP, "metrics": metrics or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = CnnLstm(pretrained=False, hidden=ck["hidden"], layers=ck["layers"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
