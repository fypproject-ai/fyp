import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import time

# --- CONFIGURATION ---
MODEL_PATH = 'C:\\Users\\LENOVO\\OneDrive\\Desktop\\parking_classifier_finetuned.pth'
ANNOTATIONS_PATH = 'C:\\Users\\LENOVO\\Downloads\\Annotations (40).xml'
CAMERA_SOURCE = 'rtsp://admin:uihe@@@1214@192.168.100.16:554/Streaming/Channels/101'
INPUT_SIZE = (64, 64)
CLASS_LABELS = ["Empty", "Occupied"]
FRAME_SKIP = 2  # Process every nth frame

# --- Load Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Annotation Loader ---
def load_polygons_as_bboxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for image in root.findall("image"):
        for polygon in image.findall("polygon"):
            points = [tuple(map(float, pt.split(','))) for pt in polygon.attrib["points"].split(';')]
            x_coords, y_coords = zip(*points)
            boxes.append((int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))))
    return boxes

boxes = load_polygons_as_bboxes(ANNOTATIONS_PATH)

# --- Open RTSP Stream ---
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("❌ Could not open stream.")
    exit()
print("✅ Stream opened.")

# --- Processing Loop ---
frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Dropped frame.")
            time.sleep(0.5)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        for (x1, y1, x2, y2) in boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue

            try:
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                input_tensor = transform(roi_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    label = CLASS_LABELS[pred]

            except Exception as e:
                label = "Error"
                print(f"❗ Prediction Error: {e}")

            color = (0, 255, 0) if label == "Empty" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("🚘 Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
