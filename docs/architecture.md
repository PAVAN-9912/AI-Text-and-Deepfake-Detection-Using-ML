# System Architecture & Technical Specifications

This document details the complete end-to-end technical architecture and machine learning pipelines for **AI Text and Deepfake Detection Using ML**.

---

## 1. High-Level Architecture Overview

The system operates as a unified forensic inspection platform providing dual-modal verification:
1. **AI vs Human Text Analysis Engine**
2. **Real vs Deepfake Video Forensics Engine**

Both pipelines are integrated into a single-page Flask web application with asynchronous REST API communication.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Flask Web Application                         │
│                    (Templates: index.html | Static Assets)               │
└───────────────────┬─────────────────────────────────┬───────────────────┘
                    │                                 │
           POST /api/predict/text            POST /api/predict/video
                    │                                 │
                    ▼                                 ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│         Text Detection Pipeline       │ │       Video Deepfake Pipeline         │
├───────────────────────────────────────┤ ├───────────────────────────────────────┤
│ • Word TF-IDF (150k n-grams)          │ │ • OpenCV 8-Frame Uniform Extraction   │
│ • Char TF-IDF (150k n-grams)          │ │ • Face Detection (YuNet ONNX / Fallback│
│ • all-MiniLM-L6-v2 Embeddings (384-d) │ │ • ResNet-18 Spatial Features (512-d)  │
│ • Handcrafted Stylometric Features    │ │ • StandardScaler Normalization        │
│ • 194,545 Fused Feature Vector        │ │ • Temporal Conv1D (256 channels)      │
│ • Calibrated Logistic Regression      │ │ • 2-Layer BiGRU (hidden=128, bi=256)  │
│ • Decision Threshold (0.41)           │ │ • Multi-Head Attention (4 heads)      │
│ • Short-Text Safety Guard (<= 5 words)│ │ • Mean + Max Temporal Pooling         │
└───────────────────────────────────────┘ └───────────────────────────────────────┘
```

---

## 2. Text Detection Architecture

### 2.1 Feature Engineering & Fusion (194,545 Dimensions)
The text detection subsystem fuses four distinct feature spaces:

1. **Word-Level N-grams (TF-IDF)**:
   - Captures vocabulary choice, topic distribution, and lexical diversity.
   - N-gram range: $(1, 3)$ with sublinear term-frequency scaling.
   - Dimension: $100,000$ to $150,000$ active sparse features.

2. **Character-Level N-grams (TF-IDF)**:
   - Captures sub-word morphology, punctuation rhythms, and character n-gram transitions.
   - N-gram range: $(2, 6)$ with sublinear term-frequency scaling.
   - Dimension: $50,000$ to $150,000$ active sparse features.

3. **Dense Semantic Embeddings (`all-MiniLM-L6-v2`)**:
   - Captures deep contextual semantics and sentence-level coherence using a 384-dimensional pretrained transformer encoder.

4. **Stylometric & Linguistic Features**:
   - Word count, character count, sentence count, average word length, average sentence length.
   - Punctuation density, digit ratio, uppercase ratio.
   - Standardized using `StandardScaler`.

### 2.2 Classification & Decision Threshold
- **Classifier**: Logistic Regression with L2 regularization ($C=1.0$) trained on multi-source datasets (HC3, RAID, SentenceAI).
- **Calibrator**: Platt scaling calibrator (`text_calibrator.pkl`).
- **Optimal Decision Threshold**: **0.41** (determined via cross-validation to optimize F1 and balanced accuracy).
- **Short-Text Protection Rule**: Inputs with $\le 5$ words trigger a short-text safety notice to prevent false certainty on insufficient statistical evidence.

---

## 3. Video Deepfake Detection Architecture

### 3.1 Spatial-Temporal Extraction Pipeline
1. **Frame Sampling**:
   - Uniformly samples $T=8$ frames across the entire duration of the uploaded video using OpenCV (`cv2.VideoCapture`).

2. **Face Localization**:
   - Employs OpenCV YuNet ONNX face detector (`models/face_detector/face_detection_yunet.onnx`) with dynamic input resizing and score threshold of $0.5$.
   - Falls back to center-crop if no face is detected with sufficient confidence.

3. **Spatial Feature Extraction (ResNet-18)**:
   - Pretrained ResNet-18 backbone (ImageNet weights) with final fully-connected classification layer replaced by `nn.Identity()`.
   - Produces a 512-dimensional spatial descriptor per frame ($T \times 512 = 8 \times 512$).

4. **Temporal Feature Standardization**:
   - Scaled using `video_scaler.pkl` containing pre-computed feature mean and standard deviation.

### 3.2 Deep Sequence Model (`TemporalConvBiGRUModel`)
- **Temporal Convolutional Layer (Conv1D)**:
  - `nn.Conv1d(512, 256, kernel_size=3, padding=1)` captures local frame-to-frame transitional anomalies (e.g., blending boundaries, flickering).
  - Followed by `nn.BatchNorm1d(256)` and `nn.GELU()`.

- **Bidirectional GRU Sequence Processor**:
  - `nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)`.
  - Captures bidirectional long-term temporal dependencies across all sampled frames, outputting $256$-dimensional representations per step.

- **Multi-Head Self-Attention**:
  - `nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.20)`.
  - Enables the model to dynamically weight the most forensically critical frames across the sequence.
  - Residual connection with `nn.LayerNorm(256)`.

- **Temporal Pooling & Classifier Head**:
  - Dual pooling: Concatenates global average pooling and global maximum pooling ($256 + 256 = 512$).
  - Dense feed-forward layers: `Linear(512, 128)` $\to$ `GELU()` $\to$ `Dropout(0.35)` $\to$ `Linear(128, 1)`.

### 3.3 Calibration & Decision Mapping
- **Calibrator**: Isotonic Regression calibrator (`video_calibrator.pkl`).
- **Optimal Decision Threshold**: **0.32**.
- **Class Mapping**:
  - `Class 0`: Fake / Deepfake
  - `Class 1`: Real
- **Decision Logic**: If Calibrated Real Probability $\ge 0.32 \implies$ **REAL**, else **FAKE**.

---

## 4. Hardware & Runtime Optimization

- **CPU & GPU Compatible**: Automatically detects CUDA GPU when available; runs efficiently on multi-core standard CPUs.
- **Model Caching**: Backend loads models into memory once at application startup to eliminate per-request disk I/O overhead.
- **Memory Footprint**: Total disk footprint of active model weights is $< 15\text{ MB}$, well within standard hosting environments.
