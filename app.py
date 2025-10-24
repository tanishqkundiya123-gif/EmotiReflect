import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from fer import FER
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="EmotiReflect", page_icon="💎", layout="centered")

st.markdown(
    "<h1 style='text-align:center;color:#D4AF37;'>💎 EmotiReflect: AI Mirror for Emotional Awareness</h1>",
    unsafe_allow_html=True,
)
st.write("Upload a photo and let AI gently reflect your emotional state.")

@st.cache_resource
def load_detector():
    try:
        detector = FER(mtcnn=False)  # simpler, avoids TensorFlow GPU issues
    except Exception:
        detector = FER()  # fallback
    return detector

detector = load_detector()

def draw_emotions(emotions):
    fig, ax = plt.subplots()
    ax.barh(list(emotions.keys()), list(emotions.values()), color="#7d5fff")
    ax.set_xlabel("Confidence")
    st.pyplot(fig)

uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_bgr = cv2.imread(tmp.name)

    if img_bgr is None:
        st.error("Couldn't read the image. Try another one.")
    else:
        results = detector.detect_emotions(img_bgr)
        if not results:
            st.warning("No face detected. Try a clearer frontal image.")
        else:
            top_emotions = results[0]["emotions"]
            dominant = max(top_emotions, key=top_emotions.get)
            st.subheader(f"Detected Emotion: **{dominant.capitalize()}**")
            draw_emotions(top_emotions)

            messages = {
                "happy": "🌞 Keep smiling — your joy is radiant.",
                "sad": "💖 Take care of yourself — brighter moments await.",
                "angry": "🔥 Take a deep breath — calmness brings clarity.",
                "surprise": "✨ Life is full of beautiful surprises!",
                "fear": "🌙 You’re stronger than your fears.",
                "neutral": "🌼 Balance is a form of beauty."
            }
            st.markdown(f"### {messages.get(dominant, '🕊️ Stay kind to yourself.')}")
