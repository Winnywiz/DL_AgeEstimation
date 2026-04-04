import os
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = ['18-24', '25-39', '40-59', '60-plus']

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_face_crop(image_path, padding=0.4):
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(f"Missing: {image_path}")

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    results = face_detector.process(img_rgb)

    if not results.detections:
        print("No face detected! Using center crop fallback.")
        return Image.fromarray(img_rgb), False

    target_face = None
    for face in results.detections:
        target_face = face
        break 

    bbox = target_face.location_data.relative_bounding_box
    
    x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
    pad_w, pad_h = int(bw * padding), int(bh * padding)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)

    face_crop = img_rgb[y1:y2, x1:x2]
    return Image.fromarray(face_crop), True

def predict_age(image_path, model_path, device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    face_img, found = get_face_crop(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs.squeeze(), dim=0) 
        conf, idx = torch.max(probs, dim=0)

    print(f"\nPredicted age: {CLASS_NAMES[idx.item()]} with confidence of {conf.item()*100:.2f}%")
    face_img.save("last_face_detected.jpg")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    my_img = r'WIN_20260404_22_44_45_Pro.jpg'
    model = "model_best_finetuned.pth"
    
    if os.path.exists(my_img):
        predict_age(my_img, model, device)
    else:
        print(f"File not found: {my_img}")