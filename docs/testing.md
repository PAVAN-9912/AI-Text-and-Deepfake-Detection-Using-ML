# Testing & Verification Suite

This document describes the automated test suites and validation procedures for **AI Text and Deepfake Detection Using ML**.

---

## 1. Overview of Test Suites

The project includes two primary comprehensive automated test suites:

| Test Suite | File Location | Number of Checks | Description |
|:---|:---|:---:|:---|
| **API Integration Test** | `src/test_api_integration.py` | **66 checks** | Tests all Flask REST endpoints, schema validation, payload variations, error handling, and model outputs. |
| **End-to-End System Test** | `src/test_end_to_end.py` | **99 checks** | Tests full pipeline from standalone inference to Flask parity, HTML structure, JS/CSS validation, model weights, and video forensic verdicts. |

**Total Automated Checks: 165 checks (100% Pass Rate)**

---

## 2. Running Automated Tests

### 2.1 Run API Integration Suite
```bash
python src/test_api_integration.py
```

**Expected Output**:
```
============================================================
API INTEGRATION TEST SUMMARY: 66/66 checks passed
ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!
============================================================
```

### 2.2 Run End-to-End Verification Suite
```bash
python src/test_end_to_end.py
```

**Expected Output**:
```
============================================================
TEST SUMMARY: 99/99 checks passed (100.0%)
ALL TESTS PASSED! Project is ready for production.
============================================================
```

---

## 3. Test Coverage Details

### 3.1 Text Detection Invariance & Robustness
- Tests parity between standalone Python module `predict_text_final.py` and Flask `/api/predict/text`.
- Validates that difference between API output and standalone output is exactly **0.0000%**.
- Tests edge cases:
  - Empty string and whitespace-only payloads.
  - Short text inputs ($\le 5$ words) triggering safety warning.
  - Multi-paragraph human essays vs synthetic LLM generations (GPT-4, Claude, LLaMA).
  - Robustness under random character perturbations, typos, and formatting variations.

### 3.2 Video Deepfake Temporal Detection
- Validates OpenCV 8-frame extraction on standard `.mp4` test videos.
- Validates face detector initialization (`YuNet ONNX`) and graceful fallback.
- Validates ResNet-18 feature extraction and `video_scaler.pkl` standardization.
- Validates `TemporalConvBiGRUModel` forward pass and probability output.
- Verifies sample verdicts:
  - Pristine Real sample (`datasets/video/00009.mp4`): Predicts **REAL** (Real probability $\approx 69.70\%$, threshold $0.32$).
  - Deepfake sample (`datasets/video/id0_id16_0003.mp4`): Predicts **FAKE** (Fake probability $\approx 96.62\%$, threshold $0.32$).

### 3.3 Frontend & UI Verification
- Validates that `templates/index.html` contains all required DOM element IDs, navigation links, and meta tags.
- Verifies zero forbidden legacy terminology (ensures zero occurrences of `BiLSTM` or `LSTM` in UI code and styles).
- Verifies `static/css/style.css` has complete styling for cards, forensic dials, probability bars, and responsive media queries.
- Verifies `static/js/app.js` handles API calls, UI animations, clipboard copy, sample text insertion, and history state in `localStorage`.
