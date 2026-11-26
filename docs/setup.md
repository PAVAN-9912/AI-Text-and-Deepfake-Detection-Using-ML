# Setup & Installation Guide

Instructions to configure, verify, and run the **AI Content Detection Platform**.

---

## 1. System Requirements
- **OS:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python:** Python 3.11 (tested on `3.11.9`)
- **RAM:** 8 GB minimum, 16 GB recommended (~1.0–1.3 GB RSS during CPU inference)
- **PyTorch:** `torch 2.13.0+cpu` / `torchvision 0.28.0+cpu` (or CUDA build if GPU available)

---

## 2. Environment Setup

```powershell
git clone https://github.com/PAVAN-9912/AI-Text-and-Deepfake-Detection-Using-ML.git
cd AI-Text-and-Deepfake-Detection-Using-ML
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Required Frozen Model Artifacts

- **Text Forensics (`text_final_v3.4`, 194,897 features):**
  - `models/text_final/text_classifier.pkl`
  - `models/text_final/text_config.pkl`
  - `models/text_final/text_scaler.pkl`
  - `models/text_final/word_tfidf_vectorizer.pkl`
  - `models/text_final/char_tfidf_vectorizer.pkl`
  - `models/text_final/platt_calibrator.pkl`
- **Video Forensics (`PolicyC_STSeqNet_FaceViT`):**
  - `experiments/phase7f_4/checkpoints/st_seqnet_best_checkpoint.pth` (`ST-SeqNet` Epoch 8)
  - `experiments/v12_vit/checkpoints/vit_deepfake_best.pth` (`FaceViT` v12)
  - `models/face_detector/face_detection_yunet.onnx` (`YuNet` 5-point face detector)

---

## 4. Running & Verifying the Application

### Start Server
```powershell
.\venv\Scripts\python.exe app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Run Automated Verification Suites
```powershell
.\venv\Scripts\python.exe src\test_api_integration.py
.\venv\Scripts\python.exe src\test_end_to_end.py
```
