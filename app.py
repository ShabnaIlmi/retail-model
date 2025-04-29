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

st.set_page_config(page_title="Retail Object Detection", layout="wide")

# --- Install required packages ---
def install_package(package):
    st.info(f"📦 Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success(f"✅ Successfully installed {package}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to install {package}: {e}")
        return False

# Try importing YOLO, install if needed
try:
    from ultralytics import YOLO
except ImportError:
    if install_package("ultralytics"):
        try:
            from ultralytics import YOLO
        except ImportError:
            st.error("❌ Failed to import YOLO after installation")
            st.stop()

# --- Download YOLOv8n model if needed ---
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
        return True
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        return False

# Check if model exists, download if needed
if not os.path.exists(MODEL_PATH):
    if not download_model():
        st.stop()

# --- Database connection ---
@st.cache_resource
def init_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="retail_db"
        )
        st.success("✅ Database connected successfully")
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

# --- Load YOLO model ---
@st.cache_resource
def init_model():
    try:
        # Use a direct path to the model file
        model_path = os.path.abspath(MODEL_PATH)
        st.info(f"Loading model from: {model_path}")
        
        # Initialize YOLO with task explicitly defined
        model = YOLO(model_path, task='detect')
        
        # Verify model loaded properly
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Could not load the YOLO model: {str(e)}")
        st.stop()

# Initialize database and model
conn = init_db_connection()
try:
    model = init_model()
except Exception as e:
    st.error(f"❌ Model initialization failed: {str(e)}")
    st.stop()

# --- Image Processing ---
def process_image(img):
    """Process input image to ensure compatibility with YOLO"""
    # If PIL Image, convert to numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Ensure RGB format (YOLO expects RGB)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3 and isinstance(img[0,0,0], np.uint8):  # Possibly BGR
        # We'll assume BGR format from OpenCV and convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

# --- Streamlit UI ---
st.title("🛍️ Retail Object Detection App")

input_option = st.radio("Choose image input:", ["Upload from PC", "Capture from Camera"])
processed_image = None

if input_option == "Upload from PC":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            processed_image = process_image(image)
            st.image(processed_image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error processing uploaded image: {str(e)}")

elif input_option == "Capture from Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        try:
            image = Image.open(camera_input).convert('RGB')
            processed_image = process_image(image)
            st.image(processed_image, caption="Captured Image", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error processing camera image: {str(e)}")

# Run detection if image is available
if processed_image is not None:
    if st.button("🔍 Detect Objects"):
        try:
            with st.spinner("Running detection..."):
                # Run inference
                results = model(processed_image)
                
                # Get labels and detections
                labels = results[0].names
                detections = results[0].boxes.data.cpu().numpy() if hasattr(results[0].boxes, 'data') else []
                
                if len(detections) > 0:
                    # Get detected items and counts
                    detected_items = [labels[int(d[5])] for d in detections]
                    counts = Counter(detected_items)
                    
                    # Display annotated image
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="Detected Objects", use_column_width=True)
                    
                    # Display results
                    st.subheader("Detected Items:")
                    for item, count in counts.items():
                        st.write(f"🔹 {item}: {count}")
                    
                    # Save to database if connection exists
                    if conn:
                        cursor = conn.cursor()
                        try:
                            for item, count in counts.items():
                                unique_id = str(uuid.uuid4())
                                
                                # Convert image to bytes for storage
                                success, buffer = cv2.imencode('.jpg', processed_image)
                                img_bytes = buffer.tobytes() if success else None
                                
                                if img_bytes:
                                    cursor.execute("""
                                        INSERT INTO detection_data (id, item_name, quantity, image)
                                        VALUES (%s, %s, %s, %s)
                                    """, (unique_id, item, count, img_bytes))
                            
                            conn.commit()
                            st.success("✅ Detection results saved to database")
                        except Exception as e:
                            st.error(f"❌ Database error: {str(e)}")
                else:
                    st.warning("⚠️ No objects detected in the image")
        except Exception as e:
            st.error(f"❌ Detection error: {str(e)}")
else:
    st.info("📸 Please upload or capture an image to continue")