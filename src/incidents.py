"""Alert confirmation (consecutive classifier predictions) and incident recording."""
from datetime import datetime

import cv2

from . import config

DISPATCH_STATUS = "SIMULATED: emergency notification logged (nothing was actually sent)"


class IncidentManager:
    def __init__(self, db, source, mode, threshold=0.7, consecutive=3, cooldown_s=10.0):
        self.db, self.source, self.mode = db, source, mode
        self.threshold, self.consecutive, self.cooldown_s = threshold, consecutive, cooldown_s
        self.streak = 0
        self.last_incident_t = None

    @property
    def alert_active(self):
        return self.streak >= self.consecutive

    def update(self, prob, frame, frame_idx, t_sec):
        """Feed one classifier prediction. Returns a new incident id when an alert is confirmed."""
        self.streak = self.streak + 1 if prob >= self.threshold else 0
        if not self.alert_active:
            return None
        if self.last_incident_t is not None and t_sec - self.last_incident_t < self.cooldown_s:
            return None
        self.last_incident_t = t_sec
        config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.EVIDENCE_DIR / f"incident_{datetime.now():%Y%m%d_%H%M%S}_f{frame_idx}.jpg"
        cv2.imwrite(str(path), frame)
        return self.db.add(self.source, self.mode, frame_idx, t_sec, prob, path, DISPATCH_STATUS)
