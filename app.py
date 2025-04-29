import streamlit as st
import subprocess
import sys

# Ensure ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.40"])
    from ultralytics import YOLO

import numpy as np
import cv2
import uuid
import mysql.connector
from collections import Counter
from PIL import Image
import io

# --- Function to connect to MySQL ---
def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",  
            user="root",
            password="",
            database="retail_db"
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"❌ Database connection failed: {err}")
        return None

# --- Load the YOLO Model (pre-trained) ---
@st.cache_resource
def load_model():
    # Load pre-trained YOLO model from the ultralytics library (YOLOv8)
    return YOLO("yolov8n.pt")  # This uses the small version of the pre-trained YOLOv8 model, trained on COCO dataset

model = load_model()

# --- Connect to Database ---
conn = connect_to_db()
if conn:
    cursor = conn.cursor()
else:
    st.stop()

# --- Streamlit UI ---
st.title("📦 Retail Stock Detection")

# Choose Input Method
input_method = st.radio("Choose input method:", ("Upload from PC", "Capture from Camera"))

frame = None

if input_method == "Upload from PC":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

elif input_method == "Capture from Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        image = Image.open(camera_input).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# --- Process the Frame ---
if frame is not None:
    # Detect objects
    results = model.predict(frame)
    labels = results[0].names
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []

    if len(detections) > 0:
        detected_items = [labels[int(det[5])] for det in detections]
        counts = Counter(detected_items)

        # Display annotated image
        annotated_img = results[0].plot()
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Items", use_column_width=True)

        st.subheader("Detected Items")

        for item, count in counts.items():
            st.write(f"🔸 {item}: {count}")

            # Generate unique ID and convert image to bytes
            unique_id = str(uuid.uuid4())
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()

            # Insert detection data into the database
            try:
                cursor.execute("""
                    INSERT INTO detection_data (id, item_name, quantity, image)
                    VALUES (%s, %s, %s, %s)
                """, (unique_id, item, count, img_bytes))
                conn.commit()
            except mysql.connector.Error as err:
                st.error(f"❌ Failed to save data: {err}")

        st.success("✅ Detection data saved to database successfully!")
    else:
        st.warning("⚠️ No items detected in the image.")
else:
    st.info("👆 Please upload an image or capture a photo to start detection.")
