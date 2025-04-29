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
import io

# Set page configuration
st.set_page_config(page_title="Retail Object Detection", layout="wide")

# --- Check installed packages ---
def pip_list():
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
    return result.stdout

# Show installed packages for debugging
with st.expander("Show installed packages"):
    st.code(pip_list())

# --- Install required packages with specific versions ---
def install_packages():
    required_packages = [
        "ultralytics==8.0.20",  # Use a specific, stable version
        "torch>=1.7.0",
        "opencv-python>=4.1.2",
        "numpy>=1.18.5",
        "pillow>=7.1.2"
    ]
    
    for package in required_packages:
        st.info(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            st.success(f"✅ Installed {package}")
        except Exception as e:
            st.warning(f"⚠️ Issue installing {package}: {e}")

# Install required packages
with st.spinner("Installing required packages..."):
    install_packages()

# Download model if not exists
MODEL_PATH = "yolov8n.pt"  # Using the smaller model for faster loading
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

# Download model file
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info(f"📥 Downloading YOLOv8 model file...")
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

# --- Alternative model loading (without caching) ---
def load_model():
    st.info("🔄 Loading YOLO model...")
    
    # Force reload of ultralytics after installation
    import importlib
    if 'ultralytics' in sys.modules:
        importlib.reload(sys.modules['ultralytics'])
    
    try:
        # Import YOLO after potential reinstallation
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error(f"Detailed error type: {type(e).__name__}")
        st.stop()

# --- Process image ---
def process_image(image):
    """Process image for model input"""
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convert color format if needed
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # Assume BGR and convert to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_np
    
    return image_rgb

# --- Connect to database ---
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
        st.warning(f"⚠️ Database connection failed: {str(e)}")
        return None

# --- Main UI ---
st.title("🛍️ Retail Object Detection App")

# Input selection
input_option = st.radio("Choose image input:", ["Upload from PC", "Capture from Camera"])

# Try to load model
try:
    # Import here after installation
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.error("Trying to reinstall ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "ultralytics"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.20"])
    try:
        # Re-import after reinstallation
        import importlib
        if 'ultralytics' in sys.modules:
            importlib.reload(sys.modules['ultralytics'])
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        st.success("✅ Model loaded successfully after reinstallation")
    except Exception as e2:
        st.error(f"❌ Still failed to load model after reinstallation: {str(e2)}")
        st.stop()

# Connect to database
db_conn = connect_to_db()

# Image source processing
if input_option == "Upload from PC":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Read image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process and detect
        if st.button("🔍 Detect Objects"):
            try:
                # Process image
                processed_image = process_image(image)
                
                # Run detection
                with st.spinner("Running detection..."):
                    results = model(processed_image)
                    
                    # Get results
                    if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'data'):
                        detections = results[0].boxes.data.cpu().numpy()
                        labels = results[0].names
                        
                        if len(detections) > 0:
                            # Process detections
                            detected_items = []
                            for det in detections:
                                cls_id = int(det[5])
                                detected_items.append(labels[cls_id])
                            
                            # Count items
                            counts = Counter(detected_items)
                            
                            # Show annotated image
                            annotated_frame = results[0].plot()
                            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                                    caption="Detected Objects", use_column_width=True)
                            
                            # Show detection counts
                            st.subheader("📦 Detected Items:")
                            for item, count in counts.items():
                                st.write(f"🔹 {item}: {count}")
                            
                            # Save to database if connected
                            if db_conn:
                                cursor = db_conn.cursor()
                                try:
                                    for item, count in counts.items():
                                        unique_id = str(uuid.uuid4())
                                        
                                        # Convert image to bytes
                                        img_byte_arr = io.BytesIO()
                                        image.save(img_byte_arr, format='JPEG')
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
                        st.error("❌ Invalid detection results")
            except Exception as e:
                st.error(f"❌ Detection failed: {str(e)}")

elif input_option == "Capture from Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        # Read image
        image = Image.open(camera_input).convert('RGB')
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Process and detect
        if st.button("🔍 Detect Objects"):
            try:
                # Process image
                processed_image = process_image(image)
                
                # Run detection
                with st.spinner("Running detection..."):
                    results = model(processed_image)
                    
                    # Get results
                    if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'data'):
                        detections = results[0].boxes.data.cpu().numpy()
                        labels = results[0].names
                        
                        if len(detections) > 0:
                            # Process detections
                            detected_items = []
                            for det in detections:
                                cls_id = int(det[5])
                                detected_items.append(labels[cls_id])
                            
                            # Count items
                            counts = Counter(detected_items)
                            
                            # Show annotated image
                            annotated_frame = results[0].plot()
                            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                                    caption="Detected Objects", use_column_width=True)
                            
                            # Show detection counts
                            st.subheader("📦 Detected Items:")
                            for item, count in counts.items():
                                st.write(f"🔹 {item}: {count}")
                            
                            # Save to database if connected
                            if db_conn:
                                cursor = db_conn.cursor()
                                try:
                                    for item, count in counts.items():
                                        unique_id = str(uuid.uuid4())
                                        
                                        # Convert image to bytes
                                        img_byte_arr = io.BytesIO()
                                        image.save(img_byte_arr, format='JPEG')
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
                        st.error("❌ Invalid detection results")
            except Exception as e:
                st.error(f"❌ Detection failed: {str(e)}")

else:
    st.info("📸 Please upload or capture an image to continue")