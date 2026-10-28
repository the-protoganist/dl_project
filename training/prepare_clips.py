"""Cut labeled intervals of source videos into fixed-length frame sequences (.npy, uint8 RGB).

Manifest CSV columns: video,label,start_sec,end_sec   (label: 1 = accident, 0 = non-accident;
start/end optional -> whole video). Build it from the TADS annotations yourself.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def sample_frames(video, start_s, end_s, step, size):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f0 = int(start_s * fps) if start_s else 0
    f1 = int(end_s * fps) if end_s else 10 ** 9
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames, i = [], f0
    while i < f1:
        ok, fr = cap.read()
        if not ok:
            break
        if (i - f0) % step == 0:
            frames.append(cv2.cvtColor(cv2.resize(fr, (size, size)), cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-stride", type=int, default=8, help="sampled frames between clip starts")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.manifest, newline="") as f:
        for n, row in enumerate(csv.DictReader(f)):
            frames = sample_frames(Path(a.video_root) / row["video"], float(row.get("start_sec") or 0),
                                   float(row["end_sec"]) if row.get("end_sec") else None,
                                   config.FRAME_STEP, config.IMG_SIZE)
            k = 0
            for s in range(0, len(frames) - config.CLIP_LEN + 1, a.window_stride):
                rel = f"clips/{Path(row['video']).stem}_{n}_{k:04d}.npy"
                np.save(out / rel, np.stack(frames[s:s + config.CLIP_LEN]))
                rows.append((rel, int(row["label"]), row["video"]))
                k += 1
            print(f"{row['video']} (label {row['label']}): {k} clips")
    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_path", "label", "source_video"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} clips -> {out / 'clips.csv'}")


if __name__ == "__main__":
    main()
