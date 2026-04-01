# YOLOv8-Face + AdaFace Production Pipeline 🚀

### A state-of-the-art face verification engine optimized for NVIDIA GPUs (RTX A5500).

Este projeto combina a detecção facial ultrarrápida do **YOLOv8** com o poder de extração de identidade do **AdaFace (IR-50)**, entregando uma solução de biometria de alta performance disponível via **FastAPI**.

## ⚡ Performance Breakdown
- **Detection (YOLOv8)**: < 5ms (A5500)
- **Extraction (AdaFace)**: < 3ms (A5500)
- **Total Pipeline**: **4.8ms per face (200+ FPS)**

## 🛠️ Features
- **FastAPI Backend**: Native support for single and batch inference.
- **Batching Support**: Parallel GPU processing for high-volume requests.
- **Auto-Alignment**: Geometric face alignment based on landmarks for stable identity vectors.
- **Production Ready**: Optimized code with clean structure and robust configuration.

## 🚀 Quick Start (UV)
```bash
# Clone and setup
git clone https://github.com/victormedeiros96/yolo8face_adaface.git
cd yolo8face_adaface
uv sync

# Run the API
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

## 📊 Endpoints
- `POST /verify`: Compares two images and returns similarity (0.0 to 1.0).
- `POST /extract`: Batch processing of multiple images to get face embeddings.

## ⚖️ Thresholds
- **0.45**: Standard balanced security (Good for general usage).
- **0.60**: High security (Strict matches, lower false acceptance).

---
*Developed with ❤️ using PyTorch, Ultralytics, and CVLFace.*
