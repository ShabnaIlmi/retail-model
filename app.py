import streamlit as st
import numpy as np
import cv2
import uuid
import os
import mysql.connector
from PIL import Image
from collections import Counter
import subprocess
import sys
import requests

# --- Install ultralytics if not available ---
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.40"])
    from ultralytics import YOLO

# --- Download YOLOv8n.pt manually if not found ---
MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

def download_model():
    st.info("📥 Downloading YOLOv8n model file...")
    try:
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

if not os.path.exists(MODEL_PATH):
    download_model()

# --- Safe model loader without caching ---
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Could not load the YOLO model: {e}")
        st.stop()

model = load_model()

# --- MySQL Connection ---
def connect_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="retail_db"
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

conn = connect_db()
if not conn:
    st.stop()
cursor = conn.cursor()

# --- Streamlit UI ---
st.title("🛍️ Retail Object Detection App")

input_option = st.radio("Choose image input:", ["Upload from PC", "Capture from Camera"])
frame = None

if input_option == "Upload from PC":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

elif input_option == "Capture from Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        image = Image.open(camera_input).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

if frame is not None:
    results = model.predict(frame)
    labels = results[0].names
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []

    if len(detections) > 0:
        detected_items = [labels[int(d[5])] for d in detections]
        counts = Counter(detected_items)

        annotated_img = results[0].plot()
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)

        st.subheader("Detected Items:")
        for item, count in counts.items():
            st.write(f"🔹 {item}: {count}")

            try:
                # Save image as bytes
                unique_id = str(uuid.uuid4())
                _, buffer = cv2.imencode('.jpg', frame)
                img_bytes = buffer.tobytes()

                cursor.execute("""
                    INSERT INTO detection_data (id, item_name, quantity, image)
                    VALUES (%s, %s, %s, %s)
                """, (unique_id, item, count, img_bytes))
                conn.commit()
            except mysql.connector.Error as e:
                st.error(f"❌ Error saving to DB: {e}")
        st.success("✅ All detection results saved to the database.")
    else:
        st.warning("⚠️ No objects detected.")
else:
    st.info("📸 Please upload or capture an image to continue.")
