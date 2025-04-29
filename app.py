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

# Set page configuration
st.set_page_config(page_title="Retail Object Detection", layout="wide")

# --- Install required packages if needed ---
def install_packages():
    required_packages = ["ultralytics", "opencv-python", "numpy", "pillow"]
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            st.info(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages if needed
install_packages()

# Import YOLO after ensuring it's installed
from ultralytics import YOLO

# Download model if not exists
MODEL_PATH = "yolov8m.pt"  # Using yolov8m.pt as in your Colab example
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info(f"📥 Downloading YOLOv8m model file...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Model downloaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

# Ensure model file exists
download_model()

# --- Preprocess image function (similar to your Colab code) ---
def preprocess_image(img):
    """Preprocess image for YOLO model"""
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Resize to 640x640
    img_resized = cv2.resize(img, (640, 640))
    
    # Convert to RGB (YOLO expects RGB)
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_resized  # Already RGB or grayscale
        
    return img_rgb

# --- Load the YOLO model ---
@st.cache_resource
def load_model():
    try:
        # Load model directly like in your Colab example
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

# --- Connect to MySQL database ---
def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="retail_db"
        )
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

# --- Main UI ---
st.title("🛍️ Retail Object Detection App")

# Load model
model = load_model()

# Connect to database
db_conn = connect_to_db()

# Input selection
input_option = st.radio("Choose image input:", ["Upload from PC", "Capture from Camera"])

if input_option == "Upload from PC":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Read and display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Detect Objects"):
            # Preprocess image
            preprocessed_image = preprocess_image(np.array(image))
            
            # Run inference
            with st.spinner("Running detection..."):
                results = model.predict(preprocessed_image)
                
                # Get labels and detections
                labels = results[0].names
                detections = results[0].boxes.data.cpu().numpy()
                
                # Process detections
                detected_items = []
                if len(detections) > 0:
                    for det in detections:
                        cls_id = int(det[5])  # Get class id
                        detected_items.append(labels[cls_id])
                    
                    # Count detections
                    counts = Counter(detected_items)
                    
                    # Display annotated image
                    annotated_frame = results[0].plot()
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                             caption="Detected Objects", use_column_width=True)
                    
                    # Display counts
                    st.subheader("📦 Detected Stock Items:")
                    for item, count in counts.items():
                        st.write(f"🔹 {item}: {count}")
                    
                    # Save to database if connected
                    if db_conn:
                        cursor = db_conn.cursor()
                        try:
                            for item, count in counts.items():
                                unique_id = str(uuid.uuid4())
                                
                                # Convert image to bytes
                                img_pil = Image.fromarray(np.array(image))
                                img_byte_arr = io.BytesIO()
                                img_pil.save(img_byte_arr, format='JPEG')
                                img_bytes = img_byte_arr.getvalue()
                                
                                # Insert into database
                                cursor.execute("""
                                    INSERT INTO detection_data (id, item_name, quantity, image)
                                    VALUES (%s, %s, %s, %s)
                                """, (unique_id, item, count, img_bytes))
                            
                            db_conn.commit()
                            st.success("✅ Results saved to database")
                        except Exception as e:
                            st.error(f"❌ Database error: {str(e)}")
                else:
                    st.warning("⚠️ No objects detected")

elif input_option == "Capture from Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        # Read and display the image
        image = Image.open(camera_input).convert('RGB')
        st.image(image, caption="Captured Image", use_column_width=True)
        
        if st.button("🔍 Detect Objects"):
            # Preprocess image
            preprocessed_image = preprocess_image(np.array(image))
            
            # Run inference
            with st.spinner("Running detection..."):
                results = model.predict(preprocessed_image)
                
                # Get labels and detections
                labels = results[0].names
                detections = results[0].boxes.data.cpu().numpy()
                
                # Process detections
                detected_items = []
                if len(detections) > 0:
                    for det in detections:
                        cls_id = int(det[5])  # Get class id
                        detected_items.append(labels[cls_id])
                    
                    # Count detections
                    counts = Counter(detected_items)
                    
                    # Display annotated image
                    annotated_frame = results[0].plot()
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                             caption="Detected Objects", use_column_width=True)
                    
                    # Display counts
                    st.subheader("📦 Detected Stock Items:")
                    for item, count in counts.items():
                        st.write(f"🔹 {item}: {count}")
                    
                    # Save to database if connected
                    if db_conn:
                        cursor = db_conn.cursor()
                        try:
                            for item, count in counts.items():
                                unique_id = str(uuid.uuid4())
                                
                                # Convert image to bytes
                                img_pil = Image.fromarray(np.array(image))
                                img_byte_arr = io.BytesIO()
                                img_pil.save(img_byte_arr, format='JPEG')
                                img_bytes = img_byte_arr.getvalue()
                                
                                # Insert into database
                                cursor.execute("""
                                    INSERT INTO detection_data (id, item_name, quantity, image)
                                    VALUES (%s, %s, %s, %s)
                                """, (unique_id, item, count, img_bytes))
                            
                            db_conn.commit()
                            st.success("✅ Results saved to database")
                        except Exception as e:
                            st.error(f"❌ Database error: {str(e)}")
                else:
                    st.warning("⚠️ No objects detected")

else:
    st.info("📸 Please upload or capture an image to continue")

# Add import for BytesIO
import io