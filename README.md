# Age Group Estimator — Deployment

Streamlit web app for age group prediction using ResNet-50 + MediaPipe face detection.

## Requirements

- Docker + Docker Compose installed
- Trained model weights at `checkpoints/`

## Quick Start

```bash
# 1. Place your model weights
mkdir -p checkpoints
cp /path/to/resnet50_best_p2.pth checkpoints/

# 2. Build and run
docker compose up --build

# 3. Open in browser
# http://localhost:8501
```

## Stopping

```bash
docker compose down
```

## Rebuilding after code changes

```bash
docker compose up --build
```

## Without Docker (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Age Groups

| Class | Age Range |
|-------|-----------|
| 18-24 | 18 to 24 years |
| 25-39 | 25 to 39 years |
| 40-59 | 40 to 59 years |
| 60-plus | 60 years and above |
