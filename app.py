import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd
from fer import FER
import matplotlib.pyplot as plt
import cv2
import datetime

# --- UI config ---
st.set_page_config(page_title="EmotiReflect", page_icon="💎", layout="centered")

st.markdown(
    "<h1 style='text-align:center;color:#D4AF37;font-family:Helvetica;'>💎 EmotiReflect: AI Mirror for Emotional Awareness</h1>",
    unsafe_allow_html=True,
)
st.write("Upload a photo and receive a gentle reflection of your emotional state.")

# Create analyzer once (FER loads model)
@st.cache_resource
def load_analyzer():
    return FER(mtcnn=True)  # use MTCNN face detection for better accuracy

analyzer = load_analyzer()

# file uploader
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
log_path = "mood_log.csv"

def map_emotion_to_stress(emotion: str) -> str:
    if emotion in ["sad", "angry", "fear", "disgust"]:
        return "High Stress"
    elif emotion == "neutral":
        return "Moderate Stress"
    elif emotion in ["happy", "surprise"]:
        return "Low Stress"
    else:
        return "Unknown"

def draw_emotion_bar(emotions_dict):
    # bar plot for emotion scores
    emotions = list(emotions_dict.keys())
    scores = [emotions_dict[k] for k in emotions]
    fig, ax = plt.subplots(figsize=(6,2))
    ax.barh(emotions, scores, color="#7d5fff")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    st.pyplot(fig)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # save to temp file and run FER
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    # load using cv2 for FER
    img_bgr = cv2.imread(tmp_path)
    if img_bgr is None:
        st.error("Couldn't read the uploaded image. Try another file.")
    else:
        # FER expects RGB input as numpy array
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # analyzer.detect_emotions returns list of dicts for each face
        try:
            detections = analyzer.detect_emotions(img_rgb)
        except Exception as e:
            st.error(f"Error detecting emotions: {e}")
            detections = []

        if len(detections) == 0:
            st.warning("No face detected. Try a photo with a clear frontal face.")
        else:
            # choose largest face (most area)
            areas = []
            for d in detections:
                box = d["box"]  # x,y,w,h
                areas.append(box[2] * box[3])
            idx = int(np.argmax(areas))
            emotions = detections[idx]["emotions"]  # dict of scores
            dominant = max(emotions, key=emotions.get)

            st.subheader(f"Detected Emotion: **{dominant.capitalize()}**")
            stress_level = map_emotion_to_stress(dominant)
            if stress_level == "High Stress":
                st.markdown("### 🔴 High Stress — consider a short break, deep breaths, or talk to someone you trust.")
            elif stress_level == "Moderate Stress":
                st.markdown("### 🟠 Moderate Stress — maybe take a mindful pause or a small walk.")
            elif stress_level == "Low Stress":
                st.markdown("### 🟢 Low Stress — lovely! Keep that light energy.")
            else:
                st.markdown("### ⚪ Unable to determine stress level confidently.")

            # show emotion confidence bar
            draw_emotion_bar(emotions)

            # log timestamp + emotion
            entry = {"time": datetime.datetime.now().isoformat(timespec='seconds'), "emotion": dominant}
            df_entry = pd.DataFrame([entry])
            if os.path.exists(log_path):
                df_entry.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df_entry.to_csv(log_path, index=False)

            # show last 5 logged entries
            if os.path.exists(log_path):
                df_log = pd.read_csv(log_path)
                st.markdown("---")
                st.markdown("#### Recent Mood Log")
                st.table(df_log.tail(5).iloc[::-1].reset_index(drop=True))
