import os
import cv2
import torch
import mediapipe as mp
from PIL import Image
from torchvision import transforms

from model import ResNet50_base, ResNet50, EfficientNetB0

CLASS_NAMES = ["18-24", "25-39", "40-59", "60-plus"]
PADDING     = 0.1

ARCH_CKPT_DEFAULTS = {
    "resnet50_base"   : "./checkpoints/resnet50_base.pth",
    "resnet50"        : "./checkpoints/resnet50_finetuned.pth",
    "efficientnet_b0" : "./checkpoints/efficientnet_b0_finetuned.pth",
}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mp_face_detection = mp.solutions.face_detection
face_detector     = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def predict_image(image_path: str, model, device, padding: float = PADDING):
    """
    Detect all faces in an image, predict age group for each,
    and save annotated result.
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: could not read '{image_path}'")
        return

    h, w    = img_cv.shape[:2]
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if not results.detections:
        print("No faces detected.")
        return

    print(f"Detected {len(results.detections)} face(s). Processing...")

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x  = int(bbox.xmin  * w)
        y  = int(bbox.ymin  * h)
        bw = int(bbox.width  * w)
        bh = int(bbox.height * h)

        pad_w = int(bw * padding)
        pad_h = int(bh * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        img_tensor = preprocess(Image.fromarray(face_crop)).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs  = torch.softmax(output.squeeze(), dim=0)
            conf, idx = torch.max(probs, dim=0)
            label  = f"{CLASS_NAMES[idx.item()]} ({conf.item()*100:.1f}%)"

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = max(text_h + 5, y1 - 5)
        cv2.rectangle(img_cv, (x1, label_y - text_h - 5), (x1 + text_w, label_y + 5), (0, 255, 0), -1)
        cv2.putText(img_cv, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, img_cv)
    print(f"Saved annotated result → '{output_path}'")


def load_model(arch: str, weights_path: str, device):
    if arch == "resnet50_base":
        model = ResNet50_base(num_classes=len(CLASS_NAMES))
    elif arch == "resnet50":
        model, _ = ResNet50(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    elif arch == "efficientnet_b0":
        model, _ = EfficientNetB0(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device).eval()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Age group prediction from image")
    parser.add_argument("image",    type=str,  help="Path to input image")
    parser.add_argument("--arch",   type=str,  default="efficientnet_b0",
                        choices=list(ARCH_CKPT_DEFAULTS.keys()),
                        help="Model architecture (default: efficientnet_b0)")
    parser.add_argument("--weights", type=str,  default=None,
                        help="Path to model weights (defaults to arch-specific checkpoint)")
    parser.add_argument("--padding", type=float, default=PADDING,
                        help="Face crop padding ratio")
    args = parser.parse_args()

    weights_path = args.weights or ARCH_CKPT_DEFAULTS[args.arch]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    model = load_model(args.arch, weights_path, device)
    print(f"Loaded {args.arch} weights from '{weights_path}'")

    if not os.path.exists(args.image):
        print(f"Error: image not found at '{args.image}'")
    else:
        predict_image(args.image, model, device, padding=args.padding)
