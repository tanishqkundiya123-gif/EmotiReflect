import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="EmotiReflect", page_icon="💎", layout="centered")

st.markdown("<h1 style='text-align:center;color:#D4AF37;'>💎 EmotiReflect: The AI Mirror for Emotional Awareness</h1>", unsafe_allow_html=True)
st.write("Upload a photo and let AI reflect your emotional state with empathy and elegance.")

uploaded_file = st.file_uploader("Upload your image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        result = DeepFace.analyze(img_path=tmp.name, actions=['emotion'], enforce_detection=False)

    if isinstance(result, list):
        result = result[0]

    emotion = result.get("dominant_emotion", "unknown")
    st.subheader(f"Detected Emotion: **{emotion.capitalize()}**")

    responses = {
        "happy": "🌞 You look radiant today! Keep sharing that joy.",
        "sad": "💖 It's okay to feel down sometimes. Remember: every storm passes.",
        "angry": "🔥 Take a breath. Calmness is your true power.",
        "fear": "🌙 Courage lives within you; you’re stronger than you think.",
        "surprise": "✨ Life’s little surprises keep it beautiful, don’t they?",
        "neutral": "🌼 A moment of calm — stay balanced and grounded."
    }

    st.markdown(f"### {responses.get(emotion, '🕊️ Be kind to yourself today.')}")
    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Designed with 💎 perfection and Tanishq elegance ✨</p>", unsafe_allow_html=True)


