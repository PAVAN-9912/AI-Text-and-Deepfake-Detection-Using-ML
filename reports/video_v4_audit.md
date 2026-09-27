# Video Deepfake Detector V4 — Comprehensive Subsystem Audit

**Project**: AI Content Detection — Final-Year Major Project  
**Date**: September 27, 2026  
**Audited Subsystem**: Video AI / Deepfake Detection Pipeline  
**Production Baseline Directory**: [`models/video_final/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/video_final/)  
**Inference Script**: [`src/predict_video_final.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/predict_video_final.py)  

---

## 1. Executive Summary & Audit Overview

This audit covers the entirety of the video detection subsystem in the workspace, detailing the data splits, face-aware feature pipelines, temporal modeling architectures, probability calibration routines, and official Celeb-DF v2 benchmark evaluations.

```
Input Video (.mp4, .avi, .mov, etc.)
 │
 ├── OpenCV Uniform Temporal Sampling: 8 frames
 ├── YuNet ONNX Face Detection (Score Thresh=0.6, NMS=0.3, Top-1 Face)
 ├── Dynamic 25% Margin Face Cropping & Aspect-Preserving Normalization
 ├── ImageNet Pretrained ResNet-18 Feature Extractor (512-D per frame)
 ├── Feature Standardization (Z-score normalized on Train split)
 │
 └── FaceSpatialTemporalModel (BiGRU + Multihead Attention + Dual Pooling)
      │
      ├── Dense Projection (512 -> 256) + LayerNorm + Dropout(0.25)
      ├── 2-Layer Bidirectional GRU (Hidden=128 -> Output=256)
      ├── 4-Head Temporal Multihead Attention (Embed=256, Dropout=0.20)
      ├── Residual Add & LayerNorm (gru_out + attn_out)
      ├── Dual Temporal Pooling (Mean Pool + Max Pool = 512-D)
      └── Multi-Layer Perceptron (512 -> 128 -> 1)
           │
           └── Platt Sigmoid Calibrator (Fitted on Validation Split)
                │
                └── Decision Threshold (t = 0.15 on P(Real))
```

---

## 2. Dataset & Split Inventory (Celeb-DF v2)

- **Total Videos in Dataset**: $6,529$ videos
- **Feature Matrix**: [`datasets/video/face_temporal_features.csv`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/datasets/video/face_temporal_features.csv) ($485.2\text{ MB}$, $6,529\text{ rows} \times 4,103\text{ columns}$)
  - 4,096 feature columns ($8\text{ frames} \times 512\text{ ResNet-18 features}$)
  - 7 metadata columns (`video_id`, `filename`, `split`, `label`, `label_name`, `frame_count`, `face_detected_frames`)

### Exact Split Partitioning:

| Split Partition | Total Videos | Real (Label 1) | Fake (Label 0) | Class Ratio (Fake:Real) | Overlap with Test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | $4,810$ | $570$ | $4,240$ | $7.44 : 1$ | **0.0% (Zero Leakage)** |
| **Validation Split** | $1,201$ | $142$ | $1,059$ | $7.46 : 1$ | **0.0% (Zero Leakage)** |
| **Official Benchmark Test** | $518$ | $178$ | $340$ | $1.91 : 1$ | **Frozen Benchmark** |
| **Total** | **6,529** | **890** | **5,639** | **6.34 : 1** | **Strict Zero Overlap** |

---

## 3. Production Model Architecture & Components

1. **Face Detector**:
   - Location: [`models/face_detector/face_detection_yunet.onnx`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/face_detector/face_detection_yunet.onnx)
   - Configuration: Target size $320 \times 320$, score threshold $0.6$, NMS threshold $0.3$.
   - Margin: $25\%$ expansion around bounding box with border clamping.

2. **Spatial Backbone**:
   - Pretrained ResNet-18 (`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`).
   - Global average pooling outputs a $512$-dimensional embedding per face frame.

3. **Temporal Modeling**:
   - 2-Layer Bidirectional GRU ($128$ hidden units per direction $\Rightarrow 256$-D).
   - 4-Head Multihead Attention ($256$-D embedding, dropout $0.20$).
   - Residual connection + LayerNorm.
   - Dual temporal pooling (Mean + Max $\Rightarrow 512$-D).

4. **Class Imbalance Handling**:
   - `pos_weight = 7.438` passed into `nn.BCEWithLogitsLoss`.

5. **Probability Calibration & Thresholding**:
   - Platt Sigmoid calibrator fit on validation logits.
   - Operating decision threshold $t = 0.15$ on $P(\text{Real})$ to balance sensitivity on real faces.

---

## 4. Production Performance Baseline (Official Celeb-DF v2 Test, $N=518$)

| Metric | Official Test Baseline Score |
| :--- | :---: |
| **Overall Accuracy** | **72.78%** ($377 / 518$) |
| **Balanced Accuracy** | **68.16%** |
| **ROC-AUC** | **76.52%** |
| **Real Precision** | **62.09%** |
| **Real Recall** | **53.37%** |
| **Real F1-Score** | **57.40%** |
| **Fake Precision** | **77.26%** |
| **Fake Recall** | **82.94%** |
| **Fake F1-Score** | **80.00%** |
| **Confusion Matrix** | **TN (Fake) = 282, FP = 58, FN = 83, TP (Real) = 95** |

---

## 5. Audit Conclusion & Path to V4 Improvements

The video subsystem is structurally sound, strictly isolated from test leakage, and fully functional. The primary areas for potential improvement without destroying efficiency are:
1. Exploring improved temporal pooling and attention architectures (BiLSTM, Temporal Transformer, Multi-scale attention).
2. Investigating focal loss vs class weighting formulations.
3. Feature augmentation (mild horizontal flips, scale jittering) during temporal feature training.
4. Optimal Platt calibration and dynamic thresholding.
