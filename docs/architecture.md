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

## 3. Video Deepfake & Specialist Fusion Architecture (`PolicyC_STSeqNet_FaceViT`)

### 3.1 Dual-Expert Pipeline
1. **ST-SeqNet (Full-Frame Temporal Expert, Epoch 8)**:
   - Uniformly samples $T=16$ full frames ($224 \times 224$ RGB, ImageNet normalization) using `StandardVideoPreprocessor`.
   - Extracts $1280$-d spatial embeddings via **EfficientNet-B0**, computes consecutive temporal differences $\Delta f_t = f_t - f_{t-1}$, and projects $[f_t, \Delta f_t] \in \mathbb{R}^{2560} \to \mathbb{R}^{256}$.
   - Processes sequence dynamics with a **Bidirectional GRU** ($hidden=128$, output $256$) and **Temporal Self-Attention** to produce per-frame attention weights $\alpha_t$ and calibrated AI probability (threshold **0.50**).
2. **FaceViT (Localized Facial Expert)**:
   - Samples $T=8$ frames, extracts facial regions via OpenCV YuNet ONNX (`face_detection_yunet.onnx`), and classifies cropped faces with **ViT-Base Patch16-224** (threshold **0.475**). Returns `INSUFFICIENT_EVIDENCE` on faceless media.
3. **Policy C Evidence-Gated Specialist Fusion (`StagingFusionRouter`)**:
   - **Faceless Media**: Delegates authority to ST-SeqNet (`C1_FACELESS_*`).
   - **Decisive Facial Manipulation**: Flags `FAKE` when FaceViT detects strong facial manipulation (`C2_DECISIVE_FACIAL_MANIPULATION`).
   - **Domain Conflict Protection**: Abstains with `INCONCLUSIVE` (`C3_STRONG_DOMAIN_CONFLICT`) when FaceViT reports authentic facial evidence and ST-SeqNet reports strong full-frame AI probability.

### 3.2 Verified Independent Benchmarks (Phase 7F.5 & 7F.6)
- **ST-SeqNet (Epoch 8)** on sealed $N=200$: **82.00% accuracy**, **0.8603 ROC-AUC**, **90.00% AI recall**.
- **Policy C Fusion** on sealed $N=200$ (Decided $N=172$): **83.14% decided accuracy**, **91.30% specificity**, **88.06% precision**.
- **Codec Robustness (Phase 7F.6)**: $0/40$ new false-positive flips across modern H.264 (CRF 18–38), VP9, AV1, 15 fps, and 2nd-gen transcode; $90.0\%$ genuine invariance across 10 conditions.

---

## 4. Hardware & Runtime Optimization

- **CPU & GPU Compatible**: Automatically detects CUDA GPU when available; runs efficiently on multi-core standard CPUs.
- **Model Caching**: Backend loads models into memory once at application startup to eliminate per-request disk I/O overhead.
- **Memory Footprint**: Total disk footprint of active model weights is $< 15\text{ MB}$, well within standard hosting environments.
