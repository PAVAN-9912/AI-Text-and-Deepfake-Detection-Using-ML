import os
import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet18, ResNet18_Weights
from flask import Flask, render_template, request, jsonify

# ============================================================
# PROJECT & MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEXT_MODEL_DIR = BASE_DIR / "models" / "text_final_v2_improved"
VIDEO_MODEL_DIR = BASE_DIR / "models" / "video"

TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_VIDEO_SIZE_MB = 100

# ============================================================
# FLASK APPLICATION INITIALIZATION
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_SIZE_MB * 1024 * 1024

# ============================================================
# DEVICE SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

text_word_vectorizer = None
text_char_vectorizer = None
text_linguistic_scaler = None
text_classifier = None
text_config = None
text_metrics = None
text_embedder = None
text_model_loaded = False

video_model = None
video_resnet = None
video_resnet_transforms = None
video_scaler_mean = None
video_scaler_std = None
video_config = None
video_metrics = None
video_model_loaded = False


# ============================================================
# PYTORCH TEMPORAL ATTENTION MODEL ARCHITECTURE
# ============================================================

class TemporalAttentionModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        attention_dim = hidden_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        logits = self.classifier(context).squeeze(1)
        return logits


# ============================================================
# MODEL LOADING ROUTINES (RUNS ONCE AT STARTUP)
# ============================================================

def load_text_model():
    global text_word_vectorizer, text_char_vectorizer, text_linguistic_scaler
    global text_classifier, text_config, text_metrics, text_embedder, text_model_loaded

    print("--------------------------------------------------")
    print("Loading Text Detection Model (V2 Improved)...")

    if not TEXT_MODEL_DIR.exists():
        print(f"ERROR: Text model directory not found: {TEXT_MODEL_DIR}")
        return False

    def load_pickle(name):
        path = TEXT_MODEL_DIR / name
        if not path.exists():
            print(f"Warning: Pickle file {name} not found in {TEXT_MODEL_DIR}")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    text_word_vectorizer = load_pickle("word_tfidf_vectorizer.pkl")
    text_char_vectorizer = load_pickle("char_tfidf_vectorizer.pkl")
    text_linguistic_scaler = load_pickle("linguistic_scaler.pkl")
    text_classifier = load_pickle("text_classifier.pkl")
    text_config = load_pickle("text_config.pkl")
    text_metrics = load_pickle("text_metrics.pkl")

    print(f"Loading SentenceTransformer ({TRANSFORMER_NAME})...")
    text_embedder = SentenceTransformer(TRANSFORMER_NAME)

    text_model_loaded = True
    print(f"Text Model Loaded. Feature count: {text_config.get('feature_count', 'Unknown')}")
    return True


def load_video_model():
    global video_model, video_resnet, video_resnet_transforms
    global video_scaler_mean, video_scaler_std, video_config, video_metrics, video_model_loaded

    print("--------------------------------------------------")
    print("Loading Video Deepfake Detection Model...")

    if not VIDEO_MODEL_DIR.exists():
        print(f"ERROR: Video model directory not found: {VIDEO_MODEL_DIR}")
        return False

    config_path = VIDEO_MODEL_DIR / "video_config.pkl"
    scaler_path = VIDEO_MODEL_DIR / "video_scaler.pkl"
    model_path = VIDEO_MODEL_DIR / "temporal_attention_model.pth"
    metrics_path = VIDEO_MODEL_DIR / "video_metrics.pkl"

    if not (config_path.exists() and scaler_path.exists() and model_path.exists()):
        print("ERROR: Missing required video model files.")
        return False

    with open(config_path, "rb") as f:
        video_config = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        video_scaler_mean = np.asarray(scaler_data["mean"], dtype=np.float32)
        video_scaler_std = np.asarray(scaler_data["std"], dtype=np.float32)

    if metrics_path.exists():
        with open(metrics_path, "rb") as f:
            video_metrics = pickle.load(f)

    print("Loading ResNet-18 feature extractor...")
    weights = ResNet18_Weights.DEFAULT
    video_resnet = resnet18(weights=weights)
    video_resnet.fc = nn.Identity()
    video_resnet = video_resnet.to(DEVICE)
    video_resnet.eval()
    video_resnet_transforms = weights.transforms()

    print("Loading Temporal Attention Model weights...")
    feature_dim = video_config.get("feature_dim", 512)
    hidden_dim = video_config.get("hidden_dim", 128)
    
    video_model = TemporalAttentionModel(input_dim=feature_dim, hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location=DEVICE)
    video_model.load_state_dict(state_dict)
    video_model = video_model.to(DEVICE)
    video_model.eval()

    video_model_loaded = True
    print("Video Deepfake Model Loaded Successfully.")
    return True


# Execute model loading at startup
load_text_model()
load_video_model()


# ============================================================
# INFERENCE HELPERS — TEXT
# ============================================================

def get_length_group(text):
    words = str(text).split()
    count = len(words)
    if count <= 10:
        return "very_short"
    elif count <= 30:
        return "short"
    elif count <= 60:
        return "medium"
    else:
        return "long"


def extract_linguistic_features(text):
    text_str = str(text)
    words = text_str.split()
    word_count = len(words)

    sentence_count = sum(1 for char in text_str if char in ".!?")
    if sentence_count == 0:
        sentence_count = 1

    avg_sentence_length = word_count / max(sentence_count, 1)

    if word_count > 0:
        unique_words = len(set(w.lower() for w in words))
        vocabulary_diversity = unique_words / word_count
    else:
        vocabulary_diversity = 0.0

    log_word_count = float(np.log1p(word_count))

    if word_count <= 10:
        length_bucket = 0.0
    elif word_count <= 30:
        length_bucket = 1.0
    elif word_count <= 60:
        length_bucket = 2.0
    else:
        length_bucket = 3.0

    raw_features = np.asarray([[
        log_word_count,
        sentence_count,
        avg_sentence_length,
        vocabulary_diversity,
        length_bucket
    ]], dtype=np.float32)

    return raw_features, {
        "log_word_count": round(log_word_count, 4),
        "sentence_count": int(sentence_count),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "vocabulary_diversity": round(vocabulary_diversity, 4),
        "length_bucket": float(length_bucket)
    }


def run_text_inference(text):
    if not text_model_loaded:
        raise RuntimeError("Text detection model is not loaded.")

    word_count = len(str(text).split())
    char_count = len(str(text))
    length_group = get_length_group(text)

    # 1. Word TF-IDF
    X_word = text_word_vectorizer.transform([text])

    # 2. Char TF-IDF
    X_char = text_char_vectorizer.transform([text])

    # 3. MiniLM Embedding
    X_embedding = text_embedder.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 4. Linguistic Features
    raw_ling, ling_dict = extract_linguistic_features(text)
    scaled_ling = text_linguistic_scaler.transform(raw_ling)

    # 5. Feature Fusion
    X_embed_sparse = csr_matrix(X_embedding)
    X_ling_sparse = csr_matrix(scaled_ling)

    X = hstack([X_word, X_char, X_embed_sparse, X_ling_sparse], format="csr")

    expected = text_config.get("feature_count", 194545)
    actual = X.shape[1]
    if actual != expected:
        raise ValueError(f"Feature shape mismatch: Expected {expected}, got {actual}")

    # 6. Classifier Prediction
    probabilities = text_classifier.predict_proba(X)[0]
    ai_prob = float(probabilities[1])
    human_prob = float(probabilities[0])

    threshold = float(text_config.get("decision_threshold", 0.41))
    prediction_flag = (ai_prob >= threshold)

    if prediction_flag:
        result = "AI-GENERATED"
        confidence = round(ai_prob * 100, 2)
    else:
        result = "HUMAN-WRITTEN"
        confidence = round(human_prob * 100, 2)

    return {
        "success": True,
        "media_type": "text",
        "result": result,
        "is_short_text": False,
        "confidence": confidence,
        "ai_probability": round(ai_prob * 100, 2),
        "human_probability": round(human_prob * 100, 2),
        "word_count": word_count,
        "character_count": char_count,
        "length_group": length_group,
        "threshold": round(threshold, 2),
        "linguistic_features": ling_dict
    }


# ============================================================
# INFERENCE HELPERS — VIDEO
# ============================================================

def run_video_inference(video_path):
    if not video_model_loaded:
        raise RuntimeError("Video deepfake detection model is not loaded.")

    seq_length = int(video_config.get("sequence_length", 8))
    threshold = float(video_config.get("threshold", 0.50))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    if total_frames <= seq_length:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, seq_length, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue
        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        raise ValueError("Could not extract any frames from video.")

    while len(frames) < seq_length:
        frames.append(frames[-1].copy())

    if len(frames) > seq_length:
        indices = np.linspace(0, len(frames) - 1, seq_length, dtype=int)
        frames = [frames[i] for i in indices]

    processed_tensors = [video_resnet_transforms(f) for f in frames]
    batch = torch.stack(processed_tensors).to(DEVICE)

    with torch.no_grad():
        resnet_feats = video_resnet(batch).cpu().numpy().astype(np.float32)

    standardized_feats = (resnet_feats - video_scaler_mean) / video_scaler_std
    tensor_input = torch.tensor(standardized_feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = video_model(tensor_input)
        real_prob = torch.sigmoid(logits).item()

    fake_prob = 1.0 - real_prob

    if real_prob >= threshold:
        prediction = "REAL"
        confidence = round(real_prob * 100, 2)
    else:
        prediction = "FAKE"
        confidence = round(fake_prob * 100, 2)

    return {
        "success": True,
        "media_type": "video",
        "result": prediction,
        "confidence": confidence,
        "real_probability": round(real_prob * 100, 2),
        "fake_probability": round(fake_prob * 100, 2),
        "frames_analyzed": len(frames)
    }


# ============================================================
# ROUTE HANDLERS
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "text_model": text_model_loaded,
        "video_model": video_model_loaded,
        "minilm": (text_embedder is not None),
        "feature_count": text_config.get("feature_count", 194545) if text_config else None,
        "device": str(DEVICE)
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    response = {
        "success": True,
        "text_metrics": None,
        "video_metrics": None
    }
    if text_metrics:
        response["text_metrics"] = {
            "accuracy": round(text_metrics.get("accuracy", 0.0) * 100, 2),
            "precision": round(text_metrics.get("precision", 0.0) * 100, 2),
            "recall": round(text_metrics.get("recall", 0.0) * 100, 2),
            "f1": round(text_metrics.get("f1", 0.0) * 100, 2),
            "roc_auc": round(text_metrics.get("roc_auc", 0.0) * 100, 2),
            "decision_threshold": text_metrics.get("decision_threshold", 0.41),
            "training_samples": text_metrics.get("training_samples", 9903),
            "validation_samples": text_metrics.get("validation_samples", 1089)
        }
    if video_metrics and "test" in video_metrics:
        test_m = video_metrics["test"]
        response["video_metrics"] = {
            "accuracy": round(test_m.get("accuracy", 0.0) * 100, 2),
            "balanced_accuracy": round(test_m.get("balanced_accuracy", 0.0) * 100, 2),
            "precision": round(test_m.get("precision", 0.0) * 100, 2),
            "recall": round(test_m.get("recall", 0.0) * 100, 2),
            "f1": round(test_m.get("f1", 0.0) * 100, 2),
            "roc_auc": round(test_m.get("roc_auc", 0.0) * 100, 2)
        }
    else:
        response["video_metrics"] = "Video evaluation metrics unavailable."

    return jsonify(response)


@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"success": False, "error": "No text provided."}), 400

        text = str(data["text"]).strip()
        if not text:
            return jsonify({"success": False, "error": "Empty text provided."}), 400

        words = text.split()
        word_count = len(words)

        # Application Short-Text Rule (<= 5 words)
        if word_count <= 5:
            return jsonify({
                "success": True,
                "media_type": "text",
                "result": "HUMAN-WRITTEN",
                "is_short_text": True,
                "classification_method": "short_text_rule",
                "confidence": None,
                "ai_probability": None,
                "human_probability": None,
                "word_count": word_count,
                "character_count": len(text),
                "reason": "The text is too short for reliable AI-generated text classification."
            })

        # > 5 words: Normal ML Model Pipeline
        result_payload = run_text_inference(text)
        return jsonify(result_payload)

    except Exception as e:
        print(f"Error in text analysis: {str(e)}", file=sys.stderr)
        return jsonify({"success": False, "error": f"Text processing failed: {str(e)}"}), 500


@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No video file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No video file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported video format: {ext}. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        }), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        result_payload = run_video_inference(temp_path)
        result_payload["video_filename"] = file.filename
        return jsonify(result_payload)

    except Exception as e:
        print(f"Error in video analysis: {str(e)}", file=sys.stderr)
        return jsonify({"success": False, "error": f"Video processing failed: {str(e)}"}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("==================================================")
    print("AI-Text-and-Deepfake-Detection-Using-ML Web App")
    print("Starting Flask server at http://localhost:5000")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)
