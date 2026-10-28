"""Streamlit dashboard: streamlit run app.py"""
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.database import IncidentDB
from src.incidents import IncidentManager
from src.video import process_video

st.set_page_config(page_title="Road Incident Detection", layout="wide")
db = IncidentDB(config.DB_PATH)

st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Mock (demo, no weights)", "Real (YOLO + CNN-LSTM)"])
mock = mode.startswith("Mock")
threshold = st.sidebar.slider("Accident probability threshold", 0.10, 0.99, 0.70, 0.01)
consecutive = st.sidebar.slider("Consecutive predictions to confirm", 1, 10, 3)
detect_every = st.sidebar.slider("Run YOLO every N frames", 1, 10, 2)
yolo_weights, ckpt_path, yolo_conf = config.YOLO_WEIGHTS, str(config.CHECKPOINT_PATH), 0.35
if not mock:
    yolo_weights = st.sidebar.text_input("YOLO weights", yolo_weights)
    ckpt_path = st.sidebar.text_input("Classifier checkpoint", ckpt_path)
    yolo_conf = st.sidebar.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)


@st.cache_resource
def load_real(weights, ckpt, conf):
    from src.classifier import CnnLstmClassifier
    from src.detector import YoloDetector
    return YoloDetector(weights, conf), CnnLstmClassifier(Path(ckpt))


st.title("Real-Time Road Incident Detection and Emergency Response")
if mock:
    st.warning("MOCK MODE: boxes come from background subtraction and accident probabilities are "
               "SIMULATED. No trained model is used; results are for demonstrating the dashboard only.")
else:
    st.info("Real mode: YOLO detects vehicles; a separate CNN-LSTM classifies accident vs. non-accident.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
run = st.button("Run analysis", type="primary", disabled=uploaded is None)

col_video, col_info = st.columns([3, 1])
video_ph = col_video.empty()
with col_info:
    prob_ph, veh_ph, streak_ph, alert_ph = st.empty(), st.empty(), st.empty(), st.empty()
progress_ph, chart_ph = st.empty(), st.empty()

if run and uploaded:
    if mock:
        from src.classifier import MockClassifier
        from src.detector import MockDetector
        detector, classifier = MockDetector(), MockClassifier()
    else:
        if not Path(ckpt_path).is_file():
            st.error(f"Checkpoint not found: {ckpt_path}. Train in Colab and place it there (see README), "
                     "or switch to Mock mode.")
            st.stop()
        try:
            detector, classifier = load_real(yolo_weights, ckpt_path, yolo_conf)
        except Exception as e:  # missing packages, bad weights, bad checkpoint
            st.error(f"Could not load real models: {e}")
            st.stop()
        classifier.reset()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
        f.write(uploaded.getvalue())
    out_path = config.OUTPUT_DIR / f"processed_{int(time.time())}.mp4"
    manager = IncidentManager(db, uploaded.name, "mock" if mock else "real", threshold, consecutive)

    probs = []
    for r in process_video(f.name, detector, classifier, manager, out_path, detect_every):
        video_ph.image(r["frame"], channels="BGR", use_container_width=True)
        p = r["prob"]
        prob_ph.metric("Accident probability", "warming up" if p is None else f"{p:.2f}")
        veh_ph.metric("Boxes in frame", r["n_vehicles"])
        streak_ph.metric("Consecutive positives", f"{r['streak']} / {consecutive}")
        if r["alert_active"]:
            alert_ph.error("ALERT: accident confirmed. Emergency response notification logged (SIMULATED).")
        else:
            alert_ph.success("No accident confirmed")
        if r["new_incident"]:
            st.toast(f"Incident #{r['new_incident']} recorded")
        if p is not None and r["idx"] % classifier.frame_step == 0:
            probs.append(p)
        if r["total"]:
            progress_ph.progress(min((r["idx"] + 1) / r["total"], 1.0))
        if probs and r["idx"] % 15 == 0:
            chart_ph.line_chart(pd.DataFrame({"P(accident)": probs}))
    if probs:
        chart_ph.line_chart(pd.DataFrame({"P(accident)": probs}))
    st.session_state["last_output"] = str(out_path)
    st.success("Analysis finished.")

last = st.session_state.get("last_output")
if last and Path(last).is_file():
    st.download_button("Download processed video (mp4v)", Path(last).read_bytes(), Path(last).name, "video/mp4")

st.divider()
st.subheader("Incident history")
rows = db.list()
if rows:
    st.dataframe(pd.DataFrame(rows).drop(columns=["evidence_path"]), use_container_width=True)
    st.subheader("Evidence frames")
    cols = st.columns(3)
    for i, row in enumerate(rows[:9]):
        if Path(row["evidence_path"]).is_file():
            cols[i % 3].image(row["evidence_path"], caption=f"#{row['id']} p={row['probability']} ({row['mode']})")
    if st.button("Clear history"):
        db.clear()
        st.rerun()
else:
    st.caption("No incidents recorded yet.")
