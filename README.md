# Age Group Estimator

Streamlit web app for age group prediction using ResNet-50 / EfficientNet-B0 + MediaPipe face detection.

## Requirements

- Python 3.10+
- Docker + Docker Compose (for containerized deployment)
- CUDA-compatible GPU recommended for training

## Quick Start (No Training Required)

Pretrained model checkpoints are included in the Docker image. Just run:

```bash
docker compose up --build
# Open http://localhost:8501
```

The app will load all available checkpoints automatically and let you select which model to use.

## Training Your Own Models

If you want to retrain the models from scratch:

### 1. Prepare the Dataset

Run the init script first to download and organize the UTKFace and FGNET datasets:

```bash
python3 init.py
```

This will download UTKFace via KaggleHub, organize images into the four age group folders (`18-24`, `25-39`, `40-59`, `60-plus`), and download FGNET.

### 2. Train a Model

Choose one of three training scripts depending on the model you want:

```bash
# ResNet-50 Baseline (frozen backbone, head-only, 10 epochs)
python3 srcs/train_base.py

# ResNet-50 Fine-Tuned (two-phase, differential LR)
python3 srcs/train_transfer.py

# EfficientNet-B0 Fine-Tuned (two-phase, differential LR)
python3 srcs/train_efficient.py
```

Trained weights will be saved to `./checkpoints/` and picked up by the app automatically.

### 3. Run the Web App

**With Docker:**
```bash
docker compose up --build
# Open http://localhost:8501
```

**Without Docker:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stopping

```bash
docker compose down
```

## Age Groups

| Class | Age Range |
|-------|-----------|
| 18-24 | 18 to 24 years |
| 25-39 | 25 to 39 years |
| 40-59 | 40 to 59 years |
| 60-plus | 60 years and above |