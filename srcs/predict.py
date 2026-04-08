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

# Setup Preprocessing (standard for ResNet)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def annotate_multiple_faces(image_path, model, device, padding):
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: Could not read {image_path}")
        return

    h, w, _ = img_cv.shape
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if not results.detections:
        print("No faces found.")
        return

    print(f"Detected {len(results.detections)} faces. Processing...")

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        
        # Calculate raw coordinates
        x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
        
        # Apply Padding
        pad_w, pad_h = int(bw * padding), int(bh * padding)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)

        # Crop and Predict
        face_crop_rgb = img_rgb[y1:y2, x1:x2]
        if face_crop_rgb.size == 0: continue
        
        pil_face = Image.fromarray(face_crop_rgb)
        img_tensor = preprocess(pil_face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs.squeeze(), dim=0)
            conf, idx = torch.max(probs, dim=0)
            age_text = f"{CLASS_NAMES[idx.item()]} ({conf.item()*100:.1f}%)"

        # Draw the face bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get text dimensions (label_size is a tuple: (width, height))
        (text_w, text_h), baseline = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Calculate label background position
        label_ymin = max(text_h, y1 - text_h - 10) 
        
        # Draw the solid green background
        cv2.rectangle(img_cv, (x1, label_ymin - text_h - 5), (x1 + text_w, label_ymin + 5), (0, 255, 0), -1)
        
        # Draw the text
        cv2.putText(img_cv, age_text, (x1, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Save the final result
    output_name = "annotated_faces.jpg"
    cv2.imwrite(output_name, img_cv)
    print(f"Success! Result saved as: {output_name}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = "model_best_finetuned.pth"

    pad = 0.1
    my_img = r'00598_AKARA.jpg'
    
    age_model = models.resnet50()
    num_ftrs = age_model.fc.in_features
    age_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    age_model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    age_model.to(device).eval()

    if os.path.exists(my_img):
        annotate_multiple_faces(my_img, age_model, device,pad)
    else:
        print(f"File not found: {my_img}")