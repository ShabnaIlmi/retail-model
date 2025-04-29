import streamlit as st
import torch
import os
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Allow YOLOv8 model loading
add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

# Define path to model
MODEL_PATH = "yolov8n.pt"

# Load YOLOv8 model safely
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning("🔄 Model file not found. Attempting to download YOLOv8n...")
        model = YOLO(MODEL_PATH)  # This will automatically download if missing
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Could not load the YOLO model: {e}")
        st.stop()

# Detect objects in the image
def detect_objects(image, model):
    results = model(image)
    return results[0].plot()  # Annotated image (NumPy array)

# Streamlit UI setup
st.set_page_config(page_title="Retail Object Detector", layout="centered")
st.title("🛒 Retail Object Detector using YOLOv8")

# Upload an image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Objects"):
        with st.spinner("🧠 Running detection..."):
            model = load_model()
            result_image = detect_objects(np.array(image), model)
            st.image(result_image, caption="📌 Detected Objects", use_container_width=True)
