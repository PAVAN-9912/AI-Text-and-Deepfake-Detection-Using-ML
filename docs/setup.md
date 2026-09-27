# Setup & Installation Guide

This guide provides step-by-step instructions to configure, install, and run **AI Text and Deepfake Detection Using ML** across Windows, macOS, and Linux environments.

---

## 1. System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python Version**: Python 3.9, 3.10, 3.11, or 3.14
- **RAM**: Minimum 8 GB recommended (16 GB for optimal speed)
- **Disk Space**: ~2 GB free disk space (for virtual environment and dependencies)
- **GPU (Optional)**: NVIDIA GPU with CUDA support for accelerated inference

---

## 2. Environment Setup

### 2.1 Clone Repository
```bash
git clone https://github.com/PAVAN-9912/AI-Text-and-Deepfake-Detection-Using-ML.git
cd AI-Text-and-Deepfake-Detection-Using-ML
```

### 2.2 Create Virtual Environment

#### On Windows (Command Prompt / PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install all required Python packages using `pip`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Verify Model Artifacts

Ensure the following pretrained model files are present:
```
models/
├── face_detector/
│   └── face_detection_yunet.onnx
├── text_final/
│   ├── char_tfidf_vectorizer.pkl
│   ├── platt_calibrator.pkl
│   ├── text_classifier.pkl
│   ├── text_scaler.pkl
│   └── word_tfidf_vectorizer.pkl
└── video_final_v4/
    ├── temporal_attention_model.pth
    ├── video_calibrator.pkl
    ├── video_metrics.pkl
    └── video_scaler.pkl
```

---

## 5. Running the Application

### Start the Flask Server
```bash
python app.py
```

### Accessing the Web Application
Open your web browser and navigate to:
```
http://localhost:5000
```
or
```
http://127.0.0.1:5000
```

---

## 6. Troubleshooting

### Issue: OpenCV `cv2` Import Error or Missing Video Codecs
**Resolution**: Ensure `opencv-python` is installed:
```bash
pip install opencv-python
```
On headless Linux servers:
```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install opencv-python-headless
```

### Issue: Torch CUDA Not Available
**Resolution**: The application automatically falls back to CPU inference. If you have an NVIDIA GPU and want CUDA acceleration, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
