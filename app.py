import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import uuid
import mysql.connector
from collections import Counter
from PIL import Image
import io

# Load the YOLO model
model = YOLO("models/retail_model.pt")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",  
    user="root",
    password="",
    database="retail_db"
)
cursor = conn.cursor()

# Streamlit UI
st.title("📦 Retail Stock Detection")

# --- NEW: Choose Input Method ---
input_method = st.radio("Choose input method:", ("Upload from PC", "Capture from Camera"))

frame = None  # Initialize frame variable

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
    detections = results[0].boxes.data.cpu().numpy()

    detected_items = [labels[int(det[5])] for det in detections]
    counts = Counter(detected_items)

    # Display image and detections
    annotated_img = results[0].plot()
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected Items", use_column_width=True)

    if counts:
        st.subheader("Detected Items")
        for item, count in counts.items():
            st.write(f"🔸 {item}: {count}")

            # Generate unique ID and convert image to bytes
            unique_id = str(uuid.uuid4())
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = buffer.tobytes()

            # Insert into database
            cursor.execute("""
                INSERT INTO detection_data (id, item_name, quantity, image)
                VALUES (%s, %s, %s, %s)
            """, (unique_id, item, count, img_bytes))

        conn.commit()
        st.success("✅ Detection data saved to database!")
    else:
        st.warning("No items detected.")
else:
    st.info("👆 Please upload an image or capture a photo to start detection.")
