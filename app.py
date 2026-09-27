import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import frozen production inference modules
try:
    from predict_text_final import (
        predict_text, extract_features_single, length_thresholds,
        word_vectorizer, classifier as text_classifier_obj, transformer_model
    )
    text_engine_ready = True
except Exception as e:
    print(f"[ERROR] Failed to load Text Detector V3.4: {e}", file=sys.stderr)
    text_engine_ready = False

try:
    from src.predict_video_final_v4 import (
        predict_video, load_video_artifacts, FACE_DETECTOR_PATH,
        MODEL_DIR as VIDEO_MODEL_DIR, DEVICE as VIDEO_DEVICE
    )
    video_engine_ready = True
except Exception as e:
    print(f"[ERROR] Failed to load Video Detector V4: {e}", file=sys.stderr)
    video_engine_ready = False

# ============================================================
# FLASK APPLICATION CONFIGURATION
# ============================================================

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_VIDEO_SIZE_MB = 100

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_SIZE_MB * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eagerly initialize video artifacts on startup
if video_engine_ready:
    try:
        load_video_artifacts()
    except Exception as e:
        print(f"[WARNING] Video artifacts preloading deferred: {e}", file=sys.stderr)


# ============================================================
# ROUTES: HOME
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# ROUTES: HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    text_ok = text_engine_ready and text_classifier_obj is not None
    video_ok = video_engine_ready and (VIDEO_MODEL_DIR / "temporal_attention_model.pth").exists()
    minilm_ok = text_engine_ready and transformer_model is not None
    face_ok = os.path.exists(FACE_DETECTOR_PATH)
    
    is_healthy = text_ok and video_ok and face_ok
    
    return jsonify({
        "status": "healthy" if is_healthy else "degraded",
        "text_model": text_ok,
        "video_model": video_ok,
        "minilm": minilm_ok,
        "face_detector": face_ok,
        "feature_count": 194897,
        "device": str(DEVICE)
    })


# ============================================================
# ROUTES: METRICS
# ============================================================

@app.route("/api/metrics", methods=["GET"])
def metrics():
    # Return frozen production benchmark metrics directly
    return jsonify({
        "success": True,
        "text_metrics": {
            "accuracy": 89.88,
            "precision": 88.98,
            "recall": 90.27,
            "f1": 89.62,
            "roc_auc": 95.63
        },
        "video_metrics": {
            "accuracy": 77.03,
            "precision": 76.50,
            "recall": 93.82,
            "f1": 84.28,
            "roc_auc": 78.33
        }
    })


# ============================================================
# ROUTES: TEXT ANALYSIS API
# ============================================================

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    try:
        if not text_engine_ready:
            return jsonify({
                "success": False,
                "error": "Text detection engine is not loaded on the server."
            }), 503

        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({
                "success": False,
                "error": "No text provided in request body."
            }), 400

        text = str(data["text"]).strip()
        if not text:
            return jsonify({
                "success": False,
                "error": "Empty text provided."
            }), 400

        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # 1. Handle Very Short Text Edge Case (1-2 words)
        if word_count <= 2:
            return jsonify({
                "success": True,
                "media_type": "text",
                "result": "INSUFFICIENT_EVIDENCE",
                "is_short_text": True,
                "limited_context": True,
                "classification_method": "short_text_rule",
                "confidence": None,
                "ai_probability": None,
                "human_probability": None,
                "word_count": word_count,
                "character_count": char_count,
                "length_group": "very_short",
                "threshold": length_thresholds.get("very_short", 0.50),
                "reason": (
                    "The input contains only 1–2 words. There is "
                    "insufficient linguistic context for reliable "
                    "AI detection."
                )
            })

        # 2. Run Frozen Production Text Detector V3.4
        res = predict_text(text)
        
        # Extract linguistic metrics for frontend accordion
        raw_scalar, _, _, _ = extract_features_single(text)
        log_word_count = round(float(raw_scalar[0, 0]), 2)
        sentence_count = int(raw_scalar[0, 1])
        avg_sentence_length = round(float(raw_scalar[0, 2]), 2)
        vocab_diversity = round(float(raw_scalar[0, 3]), 2)
        length_bucket_idx = int(raw_scalar[0, 4])
        bucket_names = ["<=10 words", "11-30 words", "31-60 words", ">60 words"]
        length_bucket_str = f"{length_bucket_idx}.0 ({bucket_names[min(length_bucket_idx, 3)]})"

        linguistic_features = {
            "log_word_count": log_word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "vocabulary_diversity": vocab_diversity,
            "length_bucket": length_bucket_str
        }

        # Format probabilities as percentages for frontend UI
        ai_prob_pct = round(res["ai_probability"] * 100, 2)
        human_prob_pct = round(res["human_probability"] * 100, 2)
        confidence_pct = round(res["confidence"] * 100, 2)

        is_ai = res["label"] == "AI Generated"
        result_label = "AI-GENERATED" if is_ai else "HUMAN-WRITTEN"
        is_short = word_count <= 30

        return jsonify({
            "success": True,
            "media_type": "text",
            "result": result_label,
            "confidence": confidence_pct,
            "ai_probability": ai_prob_pct,
            "human_probability": human_prob_pct,
            "word_count": word_count,
            "character_count": char_count,
            "length_group": res["length_group"],
            "threshold": res["threshold"],
            "is_short_text": is_short,
            "reason": "Short text heuristic applied with adapted threshold." if is_short else None,
            "linguistic_features": linguistic_features
        })

    except Exception as e:
        print(f"[ERROR] Exception in /api/analyze/text: {e}", file=sys.stderr)
        return jsonify({
            "success": False,
            "error": f"Text processing failed: {str(e)}"
        }), 500


# ============================================================
# ROUTES: VIDEO ANALYSIS API
# ============================================================

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    if not video_engine_ready:
        return jsonify({
            "success": False,
            "error": "Video deepfake detection engine is not loaded on the server."
        }), 503

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No video file uploaded."
        }), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No video file selected."
        }), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": (
                f"Unsupported video format: {ext}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            )
        }), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        # Run Frozen Production Video Detector V4 Inference Pipeline
        res = predict_video(temp_path)

        real_prob_pct = round(res["real_probability"] * 100, 2)
        fake_prob_pct = round(res["fake_probability"] * 100, 2)
        
        # Final decision corresponds to the higher probability
        if fake_prob_pct > real_prob_pct:
            result_label = "FAKE"
            confidence_pct = fake_prob_pct
        else:
            result_label = "REAL"
            confidence_pct = real_prob_pct

        return jsonify({
            "success": True,
            "media_type": "video",
            "result": result_label,
            "confidence": confidence_pct,
            "real_probability": real_prob_pct,
            "fake_probability": fake_prob_pct,
            "human_probability": real_prob_pct,
            "ai_probability": fake_prob_pct,
            "frames_analyzed": res["frames_analyzed"],
            "face_detected_frames": res["face_detected_frames"],
            "threshold": res["threshold"],
            "model": "VideoDetector_V4",
            "video_filename": file.filename
        })

    except Exception as e:
        print(f"[ERROR] Exception in /api/analyze/video: {e}", file=sys.stderr)
        return jsonify({
            "success": False,
            "error": f"Video processing failed: {str(e)}"
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI TEXT AND DEEPFAKE VIDEO DETECTION PLATFORM")
    print("=" * 60)
    print(f"Device        : {DEVICE}")
    print(f"Text Model    : TextDetector V3.4 (models/text_final/)")
    print(f"Video Model   : VideoDetector V4 (models/video_final_v4/)")
    print(f"Server URL    : http://localhost:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )