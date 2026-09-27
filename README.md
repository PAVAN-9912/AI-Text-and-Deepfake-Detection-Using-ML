# AI Text and Deepfake Detection Using ML

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-165%2F165%20passed-brightgreen.svg)]()

> **An enterprise-grade forensic detection platform for identifying AI-generated text and deepfake video manipulation using multi-modal deep learning and machine learning architectures.**

A Final Year Major Project developed at the **Department of Information Science & Engineering, Sri Krishna Institute of Technology, Bengaluru**.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Details & Algorithms](#-model-details--algorithms)
- [Performance & Benchmark Metrics](#-performance--benchmark-metrics)
- [Technology Stack](#-technology-stack)
- [Installation & Quick Start](#-installation--quick-start)
- [API Endpoints](#-api-endpoints)
- [Automated Testing & Verification](#-automated-testing--verification)
- [Project Directory Structure](#-project-directory-structure)
- [Frontend User Interface](#-frontend-user-interface)
- [Decision Logic & Thresholds](#-decision-logic--thresholds)
- [Project Team & Attribution](#-project-team--attribution)

---

## 🔍 Project Overview

The explosive rise of generative AI models (Large Language Models such as ChatGPT, Claude, and LLaMA, alongside deep generative video models such as FaceSwap, DeepFaceLab, and Diffusion models) presents significant challenges for content integrity, academic authenticity, and digital security.

**AI Text and Deepfake Detection Using ML** provides a unified forensic platform designed to analyze digital media in real-time:
1. **Text Forensics**: Distinguishes **HUMAN-WRITTEN** vs **AI-GENERATED** text by fusing statistical n-grams, transformer sentence embeddings, and stylometric characteristics into a 194,545-dimensional feature space.
2. **Video Deepfake Forensics**: Detects synthetic facial manipulation across video sequences using an OpenCV YuNet face localization pipeline, ResNet-18 spatial feature extraction, and a deep temporal sequence architecture composed of **Temporal Conv1D + 2-layer BiGRU + Multi-Head Attention**.

---

## 🌟 Key Features

- **Dual-Modal Analysis Engine**: Unified interface capable of analyzing raw text inputs as well as video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
- **High-Dimensional Feature Fusion**: 194,545-dimensional text representation combining Word TF-IDF, Character TF-IDF, `all-MiniLM-L6-v2` dense embeddings, and stylometric features.
- **Deep Spatial-Temporal Video Modeling**: ResNet-18 spatial feature extraction paired with 1D Temporal Convolutions, 2-layer Bidirectional Gated Recurrent Units (BiGRU), and 4-head Multi-Head Self-Attention.
- **Calibrated Probabilities**: Platt scaling and Isotonic Regression ensure accurate, unskewed confidence scores.
- **Short-Text Safety Guard**: Automated short-text heuristic ($\le 5$ words) alerts users to statistical ambiguity on insufficient sample lengths.
- **Cyber-Forensics Dashboard**: Dark-themed forensic interface (`#0b0f19` slate/navy with `#38bdf8` cyan accents) featuring real-time gauge meters, probability bars, interactive history tracking via `localStorage`, and responsive glassmorphic cards.
- **Enterprise-Grade Test Suite**: 165 automated end-to-end integration and system tests with 100% pass rate.

---

## 📐 System Architecture

### 1. High-Level System Flow
```
                          ┌─────────────────────────────┐
                          │   Client Web Application    │
                          │     (Flask SPA / REST)      │
                          └──────────────┬──────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    POST /api/predict/text │                   │   POST /api/predict/video │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │  Word & Char N-gram TF-IDF│                   │  Uniform 8-Frame Sample   │
   │  MiniLM Dense Embeddings  │                   │  YuNet Face Localization  │
   │  Stylometric Statistics   │                   │  ResNet-18 Spatial Pool   │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │ Feature Fusion (194,545-d)│                   │ Temporal Conv1D (256-d)   │
   │ Calibrated LogReg (0.41)  │                   │ 2-Layer BiGRU (256-d)     │
   │                           │                   │ Multi-Head Attention (4h) │
   │                           │                   │ Calibrated Sigmoid (0.32) │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │ HUMAN vs AI Verdict       │                   │ REAL vs FAKE Verdict      │
   └───────────────────────────┘                   └───────────────────────────┘
```

### 2. Video Deepfake Sequence Architecture
```
Video Input (8 Frames)
    │
    ▼
[ResNet-18 Spatial Backbone]  ──► [8 x 512] Feature Sequence
    │
    ▼
[StandardScaler Normalization]
    │
    ▼
[Temporal Conv1D (k=3, pad=1)] ──► Local inter-frame transition features (256 channels)
    │
    ▼
[BatchNorm1d + GELU]
    │
    ▼
[2-Layer Bidirectional GRU]   ──► Bidirectional temporal representations (256-dim hidden)
    │
    ▼
[Multi-Head Attention (4 Heads)]──► Dynamic temporal frame weighting & context
    │
    ▼
[LayerNorm & Residual Connection]
    │
    ▼
[Mean Pooling ⊕ Max Pooling]   ──► Concatenated Sequence Descriptor (512-dim)
    │
    ▼
[Dense Classifier Head]        ──► Calibrated Probability & Verdict (REAL / FAKE)
```

---

## 🧠 Model Details & Algorithms

### Text Detection Pipeline
| Parameter | Configuration |
|:---|:---|
| **Word N-grams** | $(1, 3)$ n-grams, sublinear TF scaling, max features $= 100,000$ |
| **Character N-grams** | $(2, 6)$ n-grams, sublinear TF scaling, max features $= 100,000$ |
| **Dense Embeddings** | `all-MiniLM-L6-v2` transformer (384 dimensions) |
| **Linguistic Stats** | Word length, sentence length, punctuation density, digit ratio, uppercase ratio |
| **Total Features** | **194,545 fused dimensions** |
| **Classifier** | Logistic Regression with $L_2$ regularization ($C=1.0$) + Platt Calibration |
| **Decision Threshold** | **0.41** ($\ge 0.41 \implies$ AI-Generated) |

### Video Deepfake Detection Pipeline
| Parameter | Configuration |
|:---|:---|
| **Sampling Rate** | 8 frames sampled uniformly across the video timeline |
| **Face Localization** | OpenCV YuNet ONNX (`face_detection_yunet.onnx`) |
| **Spatial Backbone** | Pretrained ResNet-18 ($512$ feature outputs) |
| **Temporal Convolution** | 1D Conv ($C_{in}=512, C_{out}=256, k=3, p=1$), BatchNorm, GELU |
| **Recurrent Layer** | 2-Layer Bidirectional GRU ($hidden=128$, bidirectional output $= 256$) |
| **Attention Layer** | Multi-Head Attention ($embed\_dim=256$, $heads=4$, dropout $= 0.20$) |
| **Pooling** | Dual Mean-Pooling + Max-Pooling ($512$ concatenated dimensions) |
| **Classifier Head** | Linear($512 \to 128$) $\to$ GELU $\to$ Dropout($0.35$) $\to$ Linear($128 \to 1$) |
| **Calibration** | Isotonic Regression Calibrator |
| **Decision Threshold** | **0.32** ($\ge 0.32$ Real Probability $\implies$ REAL, else FAKE) |

---

## 📊 Performance & Benchmark Metrics

The models have been benchmarked on standard research datasets:

| Detection Module | Benchmark Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Text Classifier** | HC3 + RAID + SentenceAI | **89.26%** | 86.46% | 90.49% | **88.43%** | **96.14%** |
| **Video Deepfake Detector** | Celeb-DF v2 Benchmark | **67.37%** | 52.63% | 50.56% | **51.58%** | **65.10%** |

---

## 💻 Technology Stack

- **Backend Framework**: Python 3.9+, Flask
- **Deep Learning Framework**: PyTorch, Torchvision
- **NLP & Transformers**: HuggingFace Transformers, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Computer Vision**: OpenCV (`cv2`), YuNet ONNX, Pillow
- **Machine Learning & Math**: Scikit-Learn, SciPy, NumPy
- **Frontend Architecture**: Modern Vanilla JavaScript (ES6+), HTML5, CSS3 Custom Properties (Dark Cyber-Forensics Theme)

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/PAVAN-9912/AI-Text-and-Deepfake-Detection-Using-ML.git
cd AI-Text-and-Deepfake-Detection-Using-ML
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 🔌 API Endpoints

### 1. System Health Check
`GET /api/health`
```json
{
  "status": "healthy",
  "text_model_loaded": true,
  "video_model_loaded": true,
  "device": "cpu",
  "version": "1.0.0"
}
```

### 2. Text Detection
`POST /api/predict/text`
```json
// Request
{
  "text": "Artificial intelligence and neural networks are transforming automated forensic analysis."
}

// Response
{
  "status": "success",
  "result": "AI-GENERATED",
  "confidence": 94.25,
  "confidence_level": "High",
  "probabilities": { "ai": 94.25, "human": 5.75 },
  "threshold_used": 0.41,
  "is_short_text": false
}
```

### 3. Video Deepfake Detection
`POST /api/predict/video`  
*(Form-Data with `video` file upload)*
```json
// Response
{
  "status": "success",
  "result": "FAKE",
  "confidence": 96.62,
  "confidence_level": "High",
  "probabilities": { "fake": 96.62, "real": 3.38 },
  "video_metadata": {
    "filename": "sample_manipulated.mp4",
    "frames_analyzed": 8,
    "face_detected": true
  },
  "threshold_used": 0.32
}
```

---

## 🧪 Automated Testing & Verification

The project includes an extensive dual-suite automated testing harness with **165 individual validation checks**:

```bash
# Run REST API integration tests (66 checks)
python src/test_api_integration.py

# Run complete End-to-End system verification (99 checks)
python src/test_end_to_end.py
```

```
============================================================
TEST SUMMARY: 165/165 checks passed (100.0%)
ALL TESTS PASSED! Project is ready for production.
============================================================
```

---

## 📁 Project Directory Structure

```
AI_Content_Detection/
├── app.py                             # Flask application backend
├── requirements.txt                   # Production Python dependencies
├── README.md                          # Comprehensive documentation
├── .gitignore                         # Git exclusion rules
├── docs/                              # Technical documentation
│   ├── architecture.md                # System architecture & deep learning pipeline
│   ├── api.md                         # REST API endpoint reference
│   ├── setup.md                       # Installation and setup guide
│   └── testing.md                     # Verification & test suite documentation
├── models/                            # Trained model artifacts & weights
│   ├── face_detector/                 # YuNet ONNX face detection model
│   │   └── face_detection_yunet.onnx
│   ├── text_final/                    # Production text detection models
│   │   ├── char_tfidf_vectorizer.pkl
│   │   ├── platt_calibrator.pkl
│   │   ├── text_classifier.pkl
│   │   ├── text_scaler.pkl
│   │   └── word_tfidf_vectorizer.pkl
│   └── video_final_v4/                # Production video deepfake models
│       ├── temporal_attention_model.pth
│       ├── video_calibrator.pkl
│       ├── video_metrics.pkl
│       └── video_scaler.pkl
├── src/                               # Inference engines and test suites
│   ├── predict_video_final_v4.py      # Standalone video inference engine
│   ├── test_api_integration.py        # API integration test suite (66 checks)
│   └── test_end_to_end.py             # End-to-end verification suite (99 checks)
├── templates/
│   └── index.html                     # Main Single Page Application interface
└── static/
    ├── css/
    │   └── style.css                  # Forensic cyber-theme styles
    └── js/
        └── app.js                     # Frontend logic, animations & state
```

---

## 👥 Project Team & Attribution

**Project Title**: AI Text and Deepfake Detection Using ML  
**Institution**: Sri Krishna Institute of Technology (SKIT), Bengaluru  
**Department**: Department of Information Science & Engineering  

### Team Members
- **PAVAN S** (1KT23IS034)
- **DEEKSHITH S D** (1KT23IS020)
- **RISHI G P** (1KT23IS045)
- **SANTHOSH R** (1KT23IS049)

**Project Guide**: Mrs. Ragini Krishna, Assistant Professor, Department of ISE, SKIT Bengaluru  

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
