# AI-Text-and-Deepfake-Detection-Using-ML

> **AI-powered detection of synthetic text and deepfake video content.**

A full-stack Python machine-learning application built for a final-year major project at the Department of Information Science & Engineering, Sri Krishna Institute of Technology, Bengaluru.

---

## 🌟 Key Features

1. **AI vs Human Text Detection**:
   - Analyzes written content to classify **HUMAN-WRITTEN** vs **AI-GENERATED** text.
   - Combines Word TF-IDF (150k features), Character TF-IDF (150k features), `all-MiniLM-L6-v2` dense embeddings, and scaled linguistic statistics (**194,545 total fused features**).
   - Logistic Regression classifier with dynamic decision threshold (**0.41**).
   - Application short-text UX rule ($\le 5$ words) to prevent misleading probability scores on insufficient text.

2. **Real vs Deepfake Video Detection**:
   - Analyzes video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) for synthetic deepfake manipulation.
   - OpenCV 8-frame uniform temporal extraction.
   - Pretrained ResNet-18 spatial feature extractor (512-dim).
   - 2-layer Bidirectional LSTM with soft Temporal Attention head (`temporal_attention_model.pth`).
   - Binary verdict: **REAL** vs **FAKE / DEEPFAKE** (Class mapping: `0: Fake`, `1: Real`).

3. **Web Application & UI**:
   - Single-Page Application (SPA) powered by Python Flask.
   - Dark forensics cybersecurity aesthetic (`#0b0f19` base, `#38bdf8` cyan accent).
   - Live session history stored in browser `localStorage`.
   - Real-time system health and evaluation metric cards loaded from `text_metrics.pkl` and `video_metrics.pkl`.

---

## 📐 Architecture Diagram

### Text Detection Pipeline
```
Input Text → Preprocessing → Word TF-IDF + Char TF-IDF + MiniLM Embedding + Linguistic Features
                            ↓
                     Feature Fusion (194,545 dims)
                            ↓
                    Logistic Regression Classifier
                            ↓
                     HUMAN / AI Verdict
```

### Video Deepfake Pipeline
```
Video File → OpenCV 8-Frame Extraction → ResNet-18 Spatial Extractor (512-dim)
                                        ↓
                         Standardization & PyTorch Tensor
                                        ↓
                       2-Layer BiLSTM + Temporal Attention
                                        ↓
                              REAL / FAKE Verdict
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch & Torchvision
- Sentence-Transformers
- OpenCV (`opencv-python`)
- Scikit-Learn, SciPy, NumPy, PIL
- Flask

### Installation & Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PAVAN-9912/AI-Text-and-Deepfake-Detection-Using-ML.git
   cd AI-Text-and-Deepfake-Detection-Using-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install flask torch torchvision sentence-transformers opencv-python pillow scipy scikit-learn numpy
   ```

3. **Run the web application**:
   ```bash
   python app.py
   ```

4. **Access the Web Interface**:
   Open browser at: `http://localhost:5000`

---

## 📊 Evaluation Performance

| Module | Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Text Detector** | HC3 / RAID / SentenceAI | **89.26%** | 86.46% | 90.49% | **88.43%** | **96.14%** |
| **Video Detector** | Celeb-DF v2 | **67.37%** | 52.63% | 50.56% | **51.58%** | **65.10%** |

---

## 👥 Project Team

- **PAVAN S** (1KT23IS034)
- **DEEKSHITH S D** (1KT23IS020)
- **RISHI G P** (1KT23IS045)
- **SANTHOSH R** (1KT23IS049)

**Supervisor**: Mrs. Ragini Krishna  
*Department of Information Science & Engineering, Sri Krishna Institute of Technology, Bengaluru*
