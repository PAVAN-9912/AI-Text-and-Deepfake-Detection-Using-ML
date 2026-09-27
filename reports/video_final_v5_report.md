# Video Deepfake Detector V5 — Comprehensive Research & Optimization Report

**Project**: AI Content Detection — Final-Year Major Project  
**Date**: September 27, 2026  
**Evaluation Scope**: Video Deepfake vs. Real Face Video Detection  
**Active Production Model**: [`models/video_final_v4/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/video_final_v4/) (TemporalConvBiGRUModel)  
**V5 Experimental Checkpoint**: [`models/video_final_v5/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/video_final_v5/)  
**Inference Script**: [`src/predict_video_final_v4.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/predict_video_final_v4.py)  

---

## 1. Executive Summary & Selection Verdict

### Selection Verdict:
> **VideoDetector_V4 remains the verified production baseline.**

A rigorous research and experiment campaign was conducted to investigate whether temporal difference velocity representations, pairwise cosine temporal consistency descriptors, residual gated attention, and alternative loss formulations could improve the Real-Class Recall without damaging overall classification accuracy or ROC-AUC.

While adjusting decision thresholds and temporal representations improved Real Recall from $44.94\%$ to $48.31\%$, **VideoDetector_V4** retained superior overall performance across all primary criteria:
- **Highest Official Test Accuracy**: **77.03%** (vs. V5's 75.68%)
- **Highest Balanced Accuracy**: **69.38%** (vs. V5's 69.16%)
- **Highest ROC-AUC**: **78.33%** (vs. V5's 77.56%)
- **Highest Fake F1-Score**: **84.28%** (vs. V5's 82.93%)
- **Lowest Total Test Errors**: **119 errors** (vs. V5's 126 errors and V3 baseline's 141 errors)

---

## 2. Comprehensive Model Evolution & Benchmark Progression

Official Celeb-DF v2 Frozen Benchmark ($N=518$, $340$ Deepfakes, $178$ Real Videos):

| Evaluation Metric | Original Production Model | VideoDetector_V4 (Production Winner) | VideoDetector_V5 (Tuned Sensitivity) | Net Gain (V4 vs Original) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **72.78%** ($377/518$) | **77.03%** ($399/518$) | **75.68%** ($392/518$) | **$+4.25\%$** |
| **Balanced Accuracy** | **68.16%** | **69.38%** | **69.16%** | **$+1.22\%$** |
| **ROC-AUC** | **76.52%** | **78.33%** | **77.56%** | **$+1.81\%$** |
| **Fake Precision** | **77.26%** | **76.50%** | **76.88%** | $-0.76\%$ |
| **Fake Recall** | **82.94%** | **93.82%** | **90.00%** | **$+10.88\%$** |
| **Fake F1-Score** | **80.00%** | **84.28%** | **82.93%** | **$+4.28\%$** |
| **Real Precision** | **62.09%** | **79.21%** | **71.67%** | **$+17.12\%$** |
| **Real Recall** | **53.37%** | **44.94%** | **48.31%** | $-8.43\%$ |
| **Real F1-Score** | **57.40%** | **57.35%** | **57.72%** | $-0.05\%$ |
| **Total Test Errors** | **141** | **119** | **126** | **$-22$ Errors** |

---

## 3. V5 Experimental Architectures & Controlled Loss Variations

Evaluated on the independent validation split ($N=1,201$) and tested once on the official frozen test set ($N=518$):

| Experiment Identifier | Architectural Formulation | Loss & Weighting | Val ROC-AUC | Val BalAcc | Official Test Acc | Official Test BalAcc | Official Test ROC-AUC | Test Total Errors |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5_Exp1 (V4 Baseline, $t=0.19$)** | TemporalConv1D + BiGRU + Attn | BCE $pos\_weight=7.44$ | **75.50%** | **69.57%** | **75.68%** | **69.16%** | **77.56%** | **126** |
| **V5_Exp2 (Temporal Velocity Diff)** | Static + $\Delta x_t$ (1024D) + Cosine Sim | BCE $pos\_weight=7.44$ | **74.06%** | **69.34%** | **74.90%** | **68.30%** | **77.01%** | **130** |
| **V5_Exp4 (Temporal Velocity Focal)** | Static + $\Delta x_t$ (1024D) + Cosine Sim | Focal Loss ($\gamma=1.5$) | **75.18%** | **68.32%** | **76.06%** | **67.44%** | **76.22%** | **124** |
| **V5_Exp6 (Gated Attn $pos\_w=5.0$)** | TemporalConv + Gated BiGRU | BCE $pos\_weight=5.00$ | **75.23%** | **66.92%** | **75.68%** | **66.88%** | **75.57%** | **126** |
| **V5_Exp5 (Gated Attn $pos\_w=7.4$)** | TemporalConv + Gated BiGRU | BCE $pos\_weight=7.44$ | **74.70%** | **67.45%** | **75.48%** | **66.73%** | **76.87%** | **127** |
| **V5_Exp3 (Temporal Velocity $pos\_w=5.0$)** | Static + $\Delta x_t$ (1024D) + Cosine Sim | BCE $pos\_weight=5.00$ | **73.89%** | **68.11%** | **72.97%** | **64.56%** | **75.73%** | **140** |
| **V5_Exp7 (TemporalConv BiLSTM)** | TemporalConv + BiLSTM + Attn | BCE $pos\_weight=5.50$ | **74.62%** | **66.48%** | **73.36%** | **64.18%** | **74.95%** | **138** |

---

## 4. Error Diagnostics & Real-Class Learning Analysis

1. **Why Real Recall is Lower Than Fake Recall**:
   - The Celeb-DF v2 dataset has an intrinsic $7.44 : 1$ class imbalance in training ($4,240$ fake vs $570$ real).
   - Deepfakes display distinctive spatial and temporal anomalies (boundary blending artifacts, facial warping, inconsistent temporal noise).
   - In contrast, authentic real videos exhibit high variance in natural facial expressions, varying lighting conditions, and natural head poses.
2. **Impact of Decision Thresholding**:
   - Lowering the decision threshold from $t=0.32$ to $t=0.19$ on $P(\text{Real})$ successfully increases Real Recall from $44.94\%$ to $48.31\%$ (catching 86 real videos instead of 80).
   - However, this introduces 13 additional False Positives (fake videos misclassified as real), reducing overall accuracy from $77.03\%$ to $75.68\%$.
   - **Conclusion**: The $t=0.32$ operating point in VideoDetector_V4 strikes the optimal balance between high precision on real faces ($79.21\%$) and superior overall accuracy.

---

## 5. Architectural Specifications of the Production Model (VideoDetector_V4)

```
Input Video Stream (.mp4, .avi, .mov, etc.)
 │
 ├── 1. Uniform Temporal Frame Extraction (8 frames spanning video duration)
 ├── 2. OpenCV YuNet Face Detector (ONNX, Score Thresh=0.5, NMS=0.3)
 ├── 3. Dynamic 25% Margin Expansion & Bounding Box Clamping
 ├── 4. Face Crop Normalization: (224, 224, 3) uint8
 ├── 5. ImageNet Pretrained ResNet-18 Backbone (512-D spatial embedding per face)
 ├── 6. Z-Score Standardization (Fitted exclusively on Training Split)
 │
 └── 7. TemporalConvBiGRUModel:
      ├── Temporal 1D Convolution: Conv1d(512 -> 256, k=3, p=1) + BatchNorm1d + GELU
      ├── 2-Layer Bidirectional GRU: Hidden=128 per direction -> Output=256-D, Dropout=0.25
      ├── 4-Head Temporal Multihead Attention: Embed=256-D, Dropout=0.20
      ├── Residual Addition & LayerNorm: LayerNorm(gru_out + attn_out)
      ├── Dual Temporal Pooling: Mean Pool + Max Pool = 512-D
      └── MLP Classifier: Linear(512 -> 128) + GELU + Dropout(0.35) + Linear(128 -> 1)
           │
           └── 8. Platt Sigmoid Probability Calibrator
                │
                └── 9. Calibrated Decision Threshold (t = 0.32 on P(Real))
```

---

## 6. Mathematical Probability Integrity

The inference pipeline guarantees strict mathematical probability consistency:
$$P(\text{Deepfake}) + P(\text{Real}) \equiv 1.0000$$

- All outputs are produced via genuine Platt sigmoid scaling of the model's logits without artificial confidence padding.

---

## 7. Artifact Registry & Verification Status

- **Active Production Model**: [`models/video_final_v4/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/video_final_v4/)
- **Production Inference Pipeline**: [`src/predict_video_final_v4.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/predict_video_final_v4.py)
- **V5 Experimental Checkpoint**: [`models/video_final_v5/`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/models/video_final_v5/)
- **V5 Inference Pipeline**: [`src/predict_video_final_v5.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/predict_video_final_v5.py)
- **V5 Error Diagnostics**: [`reports/video_v5_error_analysis.json`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/reports/video_v5_error_analysis.json)
- **V5 Verification Suite**: [`src/verify_video_final_v5.py`](file:///C:/Users/PAVAN%20S/OneDrive/Desktop/AI_Content_Detection/src/verify_video_final_v5.py) (**122 / 122 automated checks passed, 100.0%**)
