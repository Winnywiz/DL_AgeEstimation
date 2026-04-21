import os
import streamlit as st
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import gdown

CLASS_NAMES     = ["18-24", "25-39", "40-59", "60-plus"]
CHECKPOINTS_DIR = "./checkpoints"
PADDING         = 0.15

CHECKPOINT_URLS = {
    "resnet50_base.pth"           : "1MQUkz9Yd2WcyNJN7mYndpIWg8fN3xky_",
    "resnet50_finetuned.pth"      : "1WZg_qlu6BzcqhS-71HQeuW5DreQNxh75",
    "efficientnet_b0_finetuned.pth" : "1VGWCbM6yNbm4za7exBqxmdVMHY-8V0BU",
}

DISPLAY_NAMES = {
    "resnet50_base.pth"           : "ResNet-50 — Baseline",
    "resnet50_finetuned.pth"      : "ResNet-50 — Fine-Tuned",
    "efficientnet_b0_finetuned.pth" : "EfficientNet-B0 - Fine-Tuned",
}

def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0f0f0f; color: #e8e8e8; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 780px !important;
    }
    hr { border: none; border-top: 1px solid #222; margin: 1.2rem 0; }

    .app-title { font-size: 1.5rem; font-weight: 700; color: #fff; margin-bottom: 0.1rem; }
    .app-sub   { font-size: 0.78rem; color: #555; margin-bottom: 1.2rem; }

    .badge {
        display: inline-flex; align-items: center; gap: 7px;
        background: #1a1a1a; border: 1px solid #2a2a2a;
        border-radius: 999px; padding: 3px 12px;
        font-size: 0.73rem; color: #888; margin-bottom: 1.2rem;
    }
    .badge-dot {
        width: 6px; height: 6px; background: #4ade80;
        border-radius: 50%; box-shadow: 0 0 4px #4ade80;
        display: inline-block;
    }
    .lbl {
        font-size: 0.65rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #444; margin-bottom: 0.4rem;
    }
    .card {
        background: #161616; border: 1px solid #222;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
    }
    .card-face { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em;
                 text-transform: uppercase; color: #444; margin-bottom: 2px; }
    .card-age  { font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 2px; }
    .card-conf { font-size: 0.73rem; color: #555; margin-bottom: 12px; }

    .bar-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
    .bar-lbl  { font-size: 0.7rem; color: #666; width: 50px; flex-shrink: 0; }
    .bar-bg   { flex: 1; height: 3px; background: #222; border-radius: 2px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 2px; background: #444; }
    .bar-fill.hi { background: #e0e0e0; }
    .bar-pct  { font-size: 0.68rem; color: #444; width: 32px; text-align: right; flex-shrink: 0; }

    .warn { background: #161616; border: 1px solid #222; border-radius: 10px;
            padding: 14px 16px; color: #555; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def ensure_checkpoints():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    for filename, file_id in CHECKPOINT_URLS.items():
        path = os.path.join(CHECKPOINTS_DIR, filename)
        if not os.path.exists(path) and file_id != "YOUR_DRIVE_FILE_ID":
            with st.spinner(f"Downloading {DISPLAY_NAMES.get(filename, filename)}..."):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)


def get_available_checkpoints(directory: str) -> list:
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.endswith(".pth")])


def build_model(checkpoint_name: str):
    name = checkpoint_name.lower()
    if "efficientnet" in name:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, len(CLASS_NAMES))
        )
    elif "base" in name:
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    else:
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        )
    return model


@st.cache_resource
def load_model(checkpoint_name: str):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(checkpoint_name)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def predict(pil_image, model, device, detector):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w    = img_rgb.shape[:2]
    results = detector.process(img_rgb)
    if not results.detections:
        return img_bgr, []
    faces = []
    for detection in results.detections:
        bbox  = detection.location_data.relative_bounding_box
        x     = int(bbox.xmin * w);   y  = int(bbox.ymin * h)
        bw    = int(bbox.width * w);  bh = int(bbox.height * h)
        pad_w = int(bw * PADDING);    pad_h = int(bh * PADDING)
        x1 = max(0, x - pad_w);       y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w);  y2 = min(h, y + bh + pad_h)
        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        tensor = preprocess(Image.fromarray(face_crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor).squeeze(), dim=0).cpu().numpy()
            idx   = int(probs.argmax())
            conf  = float(probs[idx])
        label = f"{CLASS_NAMES[idx]} ({conf*100:.1f}%)"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(th + 5, y1 - 5)
        cv2.rectangle(img_bgr, (x1, ly - th - 5), (x1 + tw + 4, ly + 5), (255, 255, 255), -1)
        cv2.putText(img_bgr, label, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        faces.append({
            "class"      : CLASS_NAMES[idx],
            "confidence" : conf,
            "probs"      : {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        })
    return img_bgr, faces

st.set_page_config(
    page_title="Age Estimator",
    page_icon="◎",
    layout="centered",
    initial_sidebar_state="collapsed"
)
inject_css()
ensure_checkpoints()

st.markdown('<div class="app-title">Age Group Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">ResNet-50 transfer learning · MediaPipe face detection · UTKFace dataset</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

checkpoints = get_available_checkpoints(CHECKPOINTS_DIR)
if not checkpoints:
    st.error("No `.pth` checkpoints found. Add Drive file IDs to `CHECKPOINT_URLS`.")
    st.stop()

st.markdown('<div class="lbl">Select Model</div>', unsafe_allow_html=True)
selected_ckpt = st.selectbox(
    "Select Model",
    options=checkpoints,
    format_func=lambda f: DISPLAY_NAMES.get(f, f),
    label_visibility="hidden"
)
st.caption(f"File: {selected_ckpt}")

st.markdown("<hr>", unsafe_allow_html=True)

model, device = load_model(selected_ckpt)
detector      = load_detector()

display_name = DISPLAY_NAMES.get(selected_ckpt, selected_ckpt)
st.markdown(
    f'<div class="badge"><span class="badge-dot"></span>{display_name} &nbsp;·&nbsp; {str(device).upper()}</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="lbl">Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded:
    pil_img = Image.open(uploaded)

    with st.spinner("Detecting faces..."):
        annotated_bgr, faces = predict(pil_img, model, device, detector)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    st.markdown('<div class="lbl">Result</div>', unsafe_allow_html=True)
    st.image(annotated_rgb, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    if not faces:
        st.markdown('<div class="warn">No faces detected — try a clearer photo.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="lbl">{len(faces)} face{"s" if len(faces) > 1 else ""} detected</div>',
            unsafe_allow_html=True
        )
        for i, face in enumerate(faces):
            bars = ""
            for cls, prob in face["probs"].items():
                hi = "hi" if cls == face["class"] else ""
                bars += (
                    f'<div class="bar-row">'
                    f'<div class="bar-lbl">{cls}</div>'
                    f'<div class="bar-bg"><div class="bar-fill {hi}" style="width:{prob*100:.1f}%"></div></div>'
                    f'<div class="bar-pct">{prob*100:.1f}%</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="card">'
                f'<div class="card-face">Face {i+1}</div>'
                f'<div class="card-age">{face["class"]}</div>'
                f'<div class="card-conf">{face["confidence"]*100:.1f}% confidence</div>'
                f'{bars}'
                f'</div>',
                unsafe_allow_html=True
            )