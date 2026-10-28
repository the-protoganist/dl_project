# Real-Time Road Incident Detection and Emergency Response

Two independent models process the same video:
1. **Vehicle detection**: pretrained YOLO (Ultralytics), boxes drawn per frame.
2. **Accident classification**: frozen MobileNetV3-Small (ImageNet) features -> trainable LSTM -> accident / non-accident.

YOLO output is never fed to the classifier. An alert is confirmed after N consecutive positive classifier predictions, then stored in SQLite with an evidence frame. Emergency dispatch is **simulated only** (logged, nothing is sent).

```
app.py                  Streamlit dashboard
src/                    models, classifier, detector, video loop, incidents, database
training/               prepare_clips.py, train.py, manifest_example.csv
notebooks/colab_training.ipynb
```

## 1. Setup (Windows PowerShell)
```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Mock mode only needs `streamlit opencv-python numpy pandas` (torch is imported only in real mode).

## 2. Mock mode (no weights, no dataset)
```powershell
streamlit run app.py
```
Choose **Mock (demo, no weights)**, upload any short video, click **Run analysis**. Boxes come from background subtraction and probabilities are simulated (a spike mid-video); this only demonstrates the dashboard, alerts, DB and evidence frames.

## 3. Training (manual, not run by this repo)
1. **Get TADS**: follow the instructions in https://github.com/cyc-gh/TADS to obtain the videos and annotations. Inspect its README for the actual folder layout and annotation format.
2. **Write `manifest.csv`** from the TADS annotations (see `training/manifest_example.csv`; its rows are placeholders). Columns: `video` (path relative to the dataset root), `label` (1 accident, 0 non-accident), `start_sec`, `end_sec` (optional; blank = whole video). Use accident intervals as label 1 and non-accident intervals/videos as label 0.
3. **Upload to Google Drive**: a folder `road_incident/` with this project's `src/` and `training/` folders plus `manifest.csv`, and the dataset videos (e.g. `TADS/`).
4. **Open `notebooks/colab_training.ipynb` in Colab**, set the runtime to GPU, edit the paths in the config cell, and run all cells. It prepares clips, trains (video-level split), prints accuracy, precision, recall, F1 and the confusion matrix on held-out test videos, and saves `accident_cnn_lstm.pt` to Drive.
5. **Download** `accident_cnn_lstm.pt` (the last cell also triggers a download) and place it at `models\accident_cnn_lstm.pt`.

Clips use 16 frames sampled every 5th video frame at 224x224 (`src/config.py`); these values are saved in the checkpoint and used at inference.

## 4. Real inference
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
Choose **Real (YOLO + CNN-LSTM)**. The checkpoint path defaults to `models\accident_cnn_lstm.pt`. The first run downloads `yolov8n.pt` automatically (internet required), or place that file in the project folder. If the checkpoint is missing, the app shows an error instead of crashing.

## Limitations
- Training code, notebook and real-mode inference were written but not run; metrics depend entirely on your training run.
- The manifest must be built manually from TADS annotations (format not verified here).
- Output video uses the `mp4v` codec; it may not play in a browser but downloads and plays in desktop players.
- Dispatch is simulated; no real emergency service is contacted.
