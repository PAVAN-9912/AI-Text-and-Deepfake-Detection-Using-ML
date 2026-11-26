# API Documentation

REST API reference for the Flask backend (`app.py`) of the **AI Content Detection Platform**.

---

## Base URL
```text
http://127.0.0.1:5000/api
```

---

## 1. System Health Check

- **Endpoint:** `GET /api/health`
- **Response (`200 OK`):**
```json
{
  "active_video_engine": "PolicyC_STSeqNet_FaceViT",
  "device": "cpu",
  "face_detector": true,
  "feature_count": 194897,
  "fusion_ready": true,
  "minilm": true,
  "status": "healthy",
  "text_model": true,
  "video_engine": "PolicyC_STSeqNet_FaceViT",
  "video_model": true,
  "vit_ready": true
}
```

---

## 2. Benchmark Metrics

- **Endpoint:** `GET /api/metrics`
- **Response (`200 OK`):** Returns validation/evaluation metrics for `text_metrics` and `video_metrics` (`PolicyC_STSeqNet_FaceViT`).

---

## 3. Text Detection Endpoint

- **Endpoint:** `POST /api/analyze/text`
- **Content-Type:** `application/json`

### Request Body
```json
{
  "text": "Your input text here..."
}
```

### Response (`200 OK`)
- **`result`:** `"HUMAN-WRITTEN"`, `"AI-GENERATED"`, or `"INSUFFICIENT_EVIDENCE"` (for `< 3` words)
- **`verdict`:** Unified verdict (`"LIKELY_HUMAN"`, `"AI_GENERATED"`, `"INCONCLUSIVE"`)
- **`ai_probability` / `human_probability` / `confidence`:** Percentages (`0–100`) or `null` when insufficient evidence
- **`word_count`**, **`length_group`** (`very_short`, `short`, `medium`, `long`), **`threshold`**, **`linguistic_features`**

### Errors
- `400 Bad Request`: Missing `"text"` key or empty text.
- `503 Service Unavailable`: Text engine not loaded.

---

## 4. Video Deepfake Detection Endpoint

- **Endpoints:** `POST /api/analyze/video` (primary) and `POST /api/predict/video` (alias)
- **Content-Type:** `multipart/form-data`
- **Form Field:** `file` (allowed extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`; max size: `50 MB`)

### Response (`200 OK`)
- **`result` / `verdict`:** `"REAL"`, `"FAKE"`, `"INCONCLUSIVE"`, or `"INSUFFICIENT_EVIDENCE"`
- **`routing_rule`:** Policy C rule (e.g., `C1_FACELESS_AI_AUTHORITY`, `C2_DECISIVE_FACIAL_MANIPULATION`, `C3_STRONG_DOMAIN_CONFLICT`, `C4_EQUIVOCAL_VIT_RESOLVED_BY_EXPD_AI`)
- **`st_seqnet_p_ai`:** Full-frame ST-SeqNet probability (`0.0–1.0`)
- **`vit_p_fake`:** FaceViT facial manipulation probability (`0.0–1.0` or `null` if `face_count == 0`)
- **`face_count`:** Number of detected aligned faces
- **`probabilities`:** `{ "real": <float|null>, "fake": <float|null> }`
- **`processing_time_ms`:** Server inference duration in milliseconds

### Errors
- `400 Bad Request`: Missing `file`, empty filename, unsupported extension, `0`-byte file, or corrupt/unreadable video.
- `413 Request Entity Too Large`: Upload exceeds `50 MB`.
- `503 Service Unavailable`: Video engine not loaded.
