# ---------------------------
# Install necessary packages (if not already)
# ---------------------------

# ---------------------------
# Import modules
# ---------------------------
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from google.colab import files
import io

# ---------------------------
# Capture image from webcam
# ---------------------------
def capture_image():
    js = Javascript('''
        async function capture() {
          const div = document.createElement('div');
          const capture = document.createElement('button');
          capture.textContent = 'Capture';
          div.appendChild(capture);

          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({video: true});

          document.body.appendChild(div);
          div.appendChild(video);
          video.srcObject = stream;
          await video.play();

          google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

          await new Promise((resolve) => capture.onclick = resolve);

          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getTracks().forEach(track => track.stop());
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          return dataUrl;
        }
        capture();
    ''')
    display(js)
    data = eval_js('capture()')
    binary = b64decode(data.split(',')[1])
    np_arr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# ---------------------------
# Upload image from computer
# ---------------------------
def upload_image():
    uploaded = files.upload()
    for fn in uploaded.keys():
        img_bytes = uploaded[fn]
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        return img

# ---------------------------
# Preprocess the image
# ---------------------------
def preprocess_image(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb

# ---------------------------
# Main
# ---------------------------

# Select input method
print("\nSelect input method:")
print("1. Real-Time Camera Detection")
print("2. Upload a Single Image from Computer")
choice = input("Enter 1 or 2: ").strip()

if choice == '1':
    frame = capture_image()
elif choice == '2':
    frame = upload_image()
else:
    print("Invalid input. Exiting.")
    exit()

# Preprocess captured/uploaded frame
preprocessed_frame = preprocess_image(frame)

# ---------------------------
# Load Retail-Specific YOLO Model
# ---------------------------
# NOTE: Make sure you download or have the correct retail model weight.
# Example for Grocery SKU or Retail Product YOLOv5 model
# Replace 'your_retail_model.pt' with actual path.

# For now I use yolov8m.pt temporarily (you can change it later to your grocery trained model)
model = YOLO('yolov8m.pt')  # <-- replace this with your retail-trained model path if available

# Save the model to file
model.save('/content/drive/MyDrive/retail-model/retail_model.pt')

# ---------------------------
# Predict on the frame
# ---------------------------
results = model.predict(preprocessed_frame)

# ---------------------------
# Process detections
# ---------------------------
labels = results[0].names
detections = results[0].boxes.data.cpu().numpy()

detected_items = []
for det in detections:
    cls_id = int(det[5])  # Get class id
    detected_items.append(labels[cls_id])

# ---------------------------
# Count and Display
# ---------------------------
counts = Counter(detected_items)

print("\n📦 Detected Stock Items:")
if counts:
    for item, count in counts.items():
        print(f"{item}: {count}")
else:
    print("No objects detected.")

# ---------------------------
# Annotate and Show Result
# ---------------------------
annotated_frame = results[0].plot()

# Save annotated image
cv2.imwrite('detection_result.jpg', annotated_frame)

# Show annotated image
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Stock Items')
plt.show()
