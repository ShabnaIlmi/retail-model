import streamlit as st
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Allow YOLOv8 model to be loaded safely
add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

# Path to the model
MODEL_PATH = "yolov8n.pt"

# Load model with caching
@st.cache_resource
def load_model():
    try:
        st.info("🔄 Loading YOLOv8 model...")
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Could not load the YOLO model: {e}")
        st.stop()

# Function to run detection
def detect_objects(image, model):
    results = model(image)
    return results[0].plot()  # Annotated image as numpy array

# Streamlit UI setup
st.set_page_config(page_title="Retail Object Detector", layout="centered")
st.title("🛒 Retail Object Detector using YOLOv8")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Process image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Objects"):
        with st.spinner("🧠 Running detection..."):
            model = load_model()
            result_image = detect_objects(np.array(image), model)
            st.image(result_image, caption="📌 Detected Objects", use_column_width=True)
