# API Documentation

This document describes the REST API endpoints exposed by the Flask backend of **AI Text and Deepfake Detection Using ML**.

---

## Base URL
```
http://localhost:5000/api
```

---

## 1. System Health Check

Returns runtime status, loaded model indicators, and active device information.

- **Endpoint**: `/health`
- **Method**: `GET`
- **Response Format**: `JSON`

### Example Response:
```json
{
  "status": "healthy",
  "text_model_loaded": true,
  "video_model_loaded": true,
  "device": "cpu",
  "version": "1.0.0"
}
```

---

## 2. Text Detection Endpoint

Classifies natural language text as either Human-Written or AI-Generated.

- **Endpoint**: `/predict/text`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Body:
```json
{
  "text": "Artificial intelligence has rapidly evolved over the past decade, revolutionizing industries and reshaping modern society."
}
```

### Response Body:
```json
{
  "status": "success",
  "result": "AI-GENERATED",
  "confidence": 94.25,
  "confidence_level": "High",
  "probabilities": {
    "ai": 94.25,
    "human": 5.75
  },
  "raw_probabilities": {
    "ai": 0.9425,
    "human": 0.0575
  },
  "metrics": {
    "word_count": 16,
    "char_count": 122,
    "sentence_count": 1,
    "avg_word_length": 6.69
  },
  "threshold_used": 0.41,
  "is_short_text": false,
  "model_version": "v1.0"
}
```

### Error Responses:
- `400 Bad Request`: When `text` field is empty or missing.
```json
{
  "status": "error",
  "message": "No text provided for analysis."
}
```

---

## 3. Video Deepfake Detection Endpoint

Analyzes an uploaded video file to detect facial manipulation and deepfake artifacts.

- **Endpoint**: `/predict/video`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

### Form Parameters:
| Parameter | Type | Required | Description |
|:---|:---|:---:|:---|
| `video` | File (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) | Yes | The video file to be analyzed. |

### Response Body:
```json
{
  "status": "success",
  "result": "FAKE",
  "confidence": 96.62,
  "confidence_level": "High",
  "probabilities": {
    "fake": 96.62,
    "real": 3.38
  },
  "raw_probabilities": {
    "fake": 0.9662,
    "real": 0.0338
  },
  "video_metadata": {
    "filename": "sample_video.mp4",
    "frames_analyzed": 8,
    "face_detected": true,
    "duration_seconds": 4.5,
    "resolution": "1920x1080"
  },
  "threshold_used": 0.32,
  "model_version": "VideoDetector_V4"
}
```

### Error Responses:
- `400 Bad Request`: When no video file is uploaded or file format is unsupported.
```json
{
  "status": "error",
  "message": "No valid video file uploaded."
}
```
- `500 Internal Server Error`: If video decoding fails or model execution encounters an error.
```json
{
  "status": "error",
  "message": "Failed to decode video frames."
}
```

---

## 4. Evaluation Metrics Endpoint

Retrieves cross-validation performance metrics and dataset evaluation scores for both models.

- **Endpoint**: `/metrics`
- **Method**: `GET`
- **Response Format**: `JSON`

### Example Response:
```json
{
  "status": "success",
  "text": {
    "accuracy": 89.26,
    "precision": 86.46,
    "recall": 90.49,
    "f1_score": 88.43,
    "roc_auc": 96.14,
    "dataset": "HC3 + RAID + SentenceAI",
    "threshold": 0.41
  },
  "video": {
    "accuracy": 67.37,
    "precision": 52.63,
    "recall": 50.56,
    "f1_score": 51.58,
    "roc_auc": 65.10,
    "dataset": "Celeb-DF v2",
    "threshold": 0.32
  }
}
```
