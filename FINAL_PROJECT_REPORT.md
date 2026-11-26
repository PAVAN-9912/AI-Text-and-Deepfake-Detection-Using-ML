# AI Content Detection Platform — Final Year Engineering Report

**Project Title:** Multi-Modal Artificial Intelligence Generated Content (AIGC) Detection Platform  
**Domain:** Natural Language Processing, Computer Vision, Deep Learning, Multi-Expert Forensics  
**Environment:** Python 3.11.9, PyTorch 2.13.0+cpu, Transformers, OpenCV YuNet, Scikit-Learn, Flask  

---

## 1. Project Overview & Problem Statement

The rapid proliferation of generative AI systems—Large Language Models (LLMs) for text and neural face-swap / diffusion video generators for visual media—poses serious risks to academic integrity, digital forensics, and information security. Single-modality detectors and naive binary video classifiers suffer from two major flaws:
1. **Modality Silos:** Separate tools are required for text and video verification.
2. **Domain Blindness & False Positives:** Facial deepfake detectors fail when no human faces are present (e.g., fully synthetic Sora/Pika/T2V clips), while full-frame detectors can be misled by social-media recompression or lack localized facial precision. Furthermore, forcing a binary `REAL`/`FAKE` guess on insufficient or conflicting evidence causes high false-positive rates.

This project delivers a unified, offline-capable web platform combining:
- **Text Forensics (`text_final_v3.4`):** A hybrid 194,897-dimensional dense + sparse + stylometric pipeline with length-adaptive thresholds and a short-text safety guard.
- **Video Forensics (`PolicyC_STSeqNet_FaceViT`):** A frozen dual-expert specialist fusion architecture combining **ST-SeqNet (Epoch 8)** (full-frame spatio-temporal detector) and **FaceViT (v12)** (aligned facial Vision Transformer) under the frozen **Policy C** router, supporting explicit `REAL`, `FAKE`, `INCONCLUSIVE`, and `INSUFFICIENT_EVIDENCE` verdicts.

---

## 2. System Architecture

```
                         ┌──────────────────────────────┐
                         │   Flask Web App (app.py)     │
                         │  Offline Mode · Max 50 MB    │
                         └──────────────┬───────────────┘
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
     ┌───────────────────────────┐             ┌───────────────────────────┐
     │  POST /api/analyze/text   │             │  POST /api/analyze/video  │
     │                           │             │  POST /api/predict/video  │
     └─────────────┬─────────────┘             └─────────────┬─────────────┘
                   ▼                                         ▼
     ┌───────────────────────────┐             ┌───────────────────────────┐
     │   Text Forensics v3.4     │             │  StandardVideoPreprocessor│
     │  - Word TF-IDF + Char     │             │  - 16 frames (224x224)    │
     │  - MiniLM-L6-v2 Dense     │             │  - YuNet Face Alignment   │
     │  - Linguistic Burstiness  │             └─────────────┬─────────────┘
     │  - 194,897 Features       │                           │
     │  - Length-Adaptive Thresh │              ┌────────────┴────────────┐
     └───────────────────────────┘              ▼                         ▼
                                     ┌──────────────────┐      ┌──────────────────┐
                                     │ ST-SeqNet (E8)   │      │  FaceViT (v12)   │
                                     │ Full-Frame Conv/ │      │ ViT-Base Patch16 │
                                     │ Temporal Attn    │      │ Threshold: 0.475 │
                                     │ Threshold: 0.50  │      └────────┬─────────┘
                                     └────────┬─────────┘               │
                                              └────────────┬────────────┘
                                                           ▼
                                          ┌────────────────────────────────┐
                                          │   Frozen Policy C Router       │
                                          │  (StagingFusionRouter)         │
                                          │  -> REAL / FAKE / INCONCLUSIVE │
                                          └────────────────────────────────┘
```

---

## 3. Model Details & Frozen Integrity

| Component | Artifact Path | Key Specs / Thresholds | Verified SHA-256 |
| :--- | :--- | :--- | :--- |
| **ST-SeqNet (Epoch 8)** | `experiments/phase7f_4/checkpoints/st_seqnet_best_checkpoint.pth` | 16-frame 224x224 full-frame spatio-temporal network; threshold `0.50` | `1d68e7418f37fd59ed09619b3bbb7787dc26479336052c7e30472c8fa835a49c` |
| **FaceViT (v12)** | `experiments/v12_vit/checkpoints/vit_deepfake_best.pth` | `google/vit-base-patch16-224` on YuNet 5-point aligned facial crops; Val AUC `0.8331`; threshold `0.475` | `89df4bdb6689290d2638c6d6dafae19916b01ce451c49df7436d534be3c8631e` |
| **Text Forensics (`text_final_v3.4`)** | `models/text_final/text_classifier.pkl` | `194,897` features; length thresholds: `very_short: 0.49`, `short: 0.53`, `medium: 0.64`, `long: 0.59`; guard `< 3` words | `654b7aee6c7f4ca85ab397ae9162705081616ab995a9017a2ac1e17dc7d1d246` |

---

## 4. Policy C Specialist Fusion Logic (`StagingFusionRouter`)

Production video inference (`src/predict_video_fusion.py`) directly invokes `StagingFusionRouter.evaluate(..., policy="POLICY_C")` (`src/experimental/phase6_4_staging/staging_fusion_router.py`):
- **`C1_FACELESS_*` (`face_count == 0`):** FaceViT is not applicable; ST-SeqNet has full authority (`REAL` if `P_ai < 0.50`, `FAKE` if `P_ai >= 0.50`).
- **`C2_DECISIVE_FACIAL_MANIPULATION` (`vit_p_fake >= 0.60`):** Decisive `FAKE` verdict regardless of full-frame score (catches localized face-swaps).
- **`C3_STRONG_DOMAIN_CONFLICT` (`vit_p_fake <= 0.20` and `st_seqnet_p_ai >= 0.80`):** Experts strongly disagree (authentic face vs. strong full-frame AI signal); returns explicit **`INCONCLUSIVE`** abstention rather than guessing.
- **`C3_DECISIVE_AUTHENTIC_FACE` (`vit_p_fake <= 0.20` and `st_seqnet_p_ai < 0.80`):** Returns `REAL`.
- **`C4_EQUIVOCAL_VIT_RESOLVED_BY_EXPD_*` (`0.20 < vit_p_fake < 0.60`):** Resolved by ST-SeqNet (`0.50` threshold) or `INCONCLUSIVE` if ST-SeqNet is unavailable.
- **`INSUFFICIENT_EVIDENCE`:** Returned when frame extraction or both experts fail to produce valid scores.

---

## 5. Strengths, Honest Limitations & Failure Modes

### Strengths
- **Calibrated Abstention:** Refuses to guess on `< 3`-word text (`INSUFFICIENT_EVIDENCE`) and strong video domain conflicts (`INCONCLUSIVE`).
- **Faceless & Face-Swap Coverage:** Handles both faceless AI generators (e.g., Pika) via ST-SeqNet and facial deepfakes (e.g., Celeb-DF synthesis) via FaceViT.
- **Offline Security & Cleanup:** Enforces `HF_HUB_OFFLINE=1`, `50 MB` upload cap (`HTTP 413`), extension/empty/corrupt checks, and `finally:` temp-file deletion.

### Honest Limitations & Observed Failure Modes
- **ST-SeqNet Domain Shift on Compressed Real Videos:** On some real Celeb-DF / YouTube-real clips, ST-SeqNet scores high (`~0.98–0.99`). When FaceViT is strongly confident (`<= 0.20`), Policy C safely abstains as `INCONCLUSIVE` (`C3_STRONG_DOMAIN_CONFLICT`). When FaceViT is equivocal (`0.3051` on `Celeb-real/id0_0000.mp4`), `C4_EQUIVOCAL_VIT_RESOLVED_BY_EXPD_AI` yields a false-positive `FAKE`.
- **Unseen Open-Source Generators:** On `EXT_AI_OPEN_COGV_001.mp4` (CogVideoX), ST-SeqNet scored `0.0004` (`REAL` false negative) because CogVideoX lies outside the training distribution.
- **CPU-Only Latency:** On CPU-only hosts (`Intel Core Ultra 5 225H`), video inference takes ~`0.62s–1.62s` per clip and process RSS sits around `1.0–1.3 GB`.
- **Minor UI Formatting Notes:** In the History panel (`#history`), `INCONCLUSIVE` video entries with `null` confidence display `null%`, and static home-page overview labels still reference the legacy `ViT-Base Patch16-224` label.

---

## 6. API & Setup Summary

- **API Reference:** See [docs/api.md](docs/api.md) (`/api/health`, `/api/metrics`, `/api/analyze/text`, `/api/analyze/video`, `/api/predict/video`).
- **Setup & Run:** See [docs/setup.md](docs/setup.md) (`.\venv\Scripts\python.exe app.py` on `http://127.0.0.1:5000`).
- **Demo Walkthrough:** See [DEMO_GUIDE.md](DEMO_GUIDE.md) and [DEMO_TEST_CASES.md](DEMO_TEST_CASES.md).

---

## 7. Automated Test & Live Verification Summary (Phase 8.2 / 8.3)

- **API Integration Suite (`src/test_api_integration.py`):** **84 / 84 checks passed (100.0%)**.
- **End-to-End System Suite (`src/test_end_to_end.py`):** **105 / 105 checks passed (100.0%)**.
- **Live Browser UI Verification (`http://127.0.0.1:5000`):** Verified Dashboard, Text Detection (empty validation, short guard, human, AI), Video Detection (`FAKE` + `C2_DECISIVE_FACIAL_MANIPULATION`, `INCONCLUSIVE` + `C3_STRONG_DOMAIN_CONFLICT`), and History (`localStorage`) with **0 console errors** and **0 leftover temp files**.

---

## 8. Future Work

- **Phase 7G Codec-Robustness Fine-Tuning:** Address ST-SeqNet H.264/social-media recompression sensitivity using paired codec augmentation (already benchmarked for 16 GB RAM readiness).
- **Expanded Diffusion Generator Coverage:** Incorporate newer open-source video generators (e.g., CogVideoX, HunyuanVideo) into training/calibration cohorts.
- **GPU / ONNX Runtime Acceleration:** Reduce video inference latency below `0.3s` per clip.
