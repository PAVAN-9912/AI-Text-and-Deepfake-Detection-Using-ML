import os
import sys
import re
import pickle
import tempfile
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer, AutoModel
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
# MINILM EMBEDDER WRAPPER (STANDALONE TRANSFORMERS ENGINE)
# ============================================================

class MiniLMEmbedder:
    """
    Direct PyTorch/HuggingFace Transformer embedder for sentence-transformers/all-MiniLM-L6-v2.
    Computes exact mean-pooled and L2-normalized 384-dimensional embeddings.
    """
    def __init__(self, model_name=TRANSFORMER_NAME, device=DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded)
                token_embeddings = model_output[0]
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy() if convert_to_numpy else result


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

    print(f"Loading MiniLM Transformer ({TRANSFORMER_NAME})...")
    text_embedder = MiniLMEmbedder(TRANSFORMER_NAME, device=DEVICE)

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
# INFERENCE HELPERS — TEXT PREPROCESSING & FEATURES
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


def split_into_sentences(text):
    """
    Split text into individual sentences using robust regex punctuation boundaries.
    """
    text_clean = str(text).strip()
    if not text_clean:
        return []

    raw_sentences = re.split(r'(?<=[.!?])\s+', text_clean)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        lines = [line.strip() for line in text_clean.split("\n") if line.strip()]
        sentences = lines if lines else [text_clean]

    return sentences


def create_sliding_windows(sentences, target_window_sentences=3, overlap=1, min_chunk_words=18):
    """
    Creates overlapping sliding windows across sentences.
    """
    if not sentences:
        return []

    if len(sentences) <= target_window_sentences:
        return [" ".join(sentences)]

    chunks = []
    step = max(1, target_window_sentences - overlap)
    total_sentences = len(sentences)

    for i in range(0, total_sentences, step):
        window = sentences[i : i + target_window_sentences]
        if not window:
            continue
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if i + target_window_sentences >= total_sentences:
            break

    if len(chunks) > 1:
        last_word_count = len(chunks[-1].split())
        if last_word_count < min_chunk_words:
            last_chunk = chunks.pop()
            if last_chunk not in chunks[-1]:
                chunks[-1] = (chunks[-1] + " " + last_chunk).strip()

    return chunks


def batch_extract_features(texts):
    """
    Extracts Word TF-IDF, Char TF-IDF, MiniLM Embeddings (batched), and Scaled Linguistic Features
    for a list of text strings matching the exact 194,545-feature training pipeline.
    """
    if not text_model_loaded:
        raise RuntimeError("Text detection model is not loaded.")

    # 1. Word TF-IDF
    X_word = text_word_vectorizer.transform(texts)

    # 2. Char TF-IDF
    X_char = text_char_vectorizer.transform(texts)

    # 3. Batched MiniLM Embedding
    X_embedding = text_embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 4. Linguistic Features
    raw_ling_list = []
    ling_dicts = []
    for t in texts:
        raw_ling, ling_dict = extract_linguistic_features(t)
        raw_ling_list.append(raw_ling[0])
        ling_dicts.append(ling_dict)

    raw_ling_matrix = np.asarray(raw_ling_list, dtype=np.float32)
    scaled_ling_matrix = text_linguistic_scaler.transform(raw_ling_matrix)

    # 5. Sparse Feature Fusion
    X_embed_sparse = csr_matrix(X_embedding)
    X_ling_sparse = csr_matrix(scaled_ling_matrix)

    X = hstack([X_word, X_char, X_embed_sparse, X_ling_sparse], format="csr")

    expected = text_config.get("feature_count", 194545)
    actual = X.shape[1]
    if actual != expected:
        raise ValueError(f"Feature shape mismatch: Expected {expected}, got {actual}")

    return X, ling_dicts


def aggregate_document_predictions(global_ai_prob, chunk_probabilities, chunk_texts, threshold=0.41):
    """
    Aggregates whole-document model prediction with sliding-window chunk probabilities.
    """
    global_ai_prob = float(global_ai_prob)
    threshold = float(threshold)

    if not chunk_probabilities:
        chunk_probabilities = [global_ai_prob]

    mean_chunk_ai_prob = float(np.mean(chunk_probabilities))
    peak_chunk_ai_prob = float(np.max(chunk_probabilities))
    ai_chunk_ratio = float(np.mean([1.0 if p >= threshold else 0.0 for p in chunk_probabilities]))
    num_chunks = len(chunk_probabilities)

    chunks_meta = []
    for idx, (p, chunk_str) in enumerate(zip(chunk_probabilities, chunk_texts)):
        chunks_meta.append({
            "chunk_index": idx + 1,
            "word_count": len(chunk_str.split()),
            "ai_probability": round(float(p) * 100, 2),
            "human_probability": round((1.0 - float(p)) * 100, 2),
            "result": "AI-GENERATED" if p >= threshold else "HUMAN-WRITTEN"
        })

    # Aggregated Classification Logic
    # Chunk-level scores are diagnostic/localization signals and are not used to override the calibrated whole-document classifier.
    if global_ai_prob >= threshold:
        result = "AI-GENERATED"
        confidence = round(global_ai_prob * 100, 2)
    else:
        result = "HUMAN-WRITTEN"
        confidence = round((1.0 - global_ai_prob) * 100, 2)

    return {
        "result": result,
        "confidence": confidence,
        "global_ai_probability": round(global_ai_prob * 100, 2),
        "mean_chunk_ai_probability": round(mean_chunk_ai_prob * 100, 2),
        "peak_chunk_ai_probability": round(peak_chunk_ai_prob * 100, 2),
        "ai_chunk_ratio": round(ai_chunk_ratio * 100, 2),
        "number_of_chunks": num_chunks,
        "chunks": chunks_meta
    }


def run_text_inference_pipeline(text):
    """
    Robust multi-tier inference pipeline supporting short text, single sentences,
    paragraphs, and long multi-sentence documents via sliding window chunking.
    """
    word_count = len(str(text).split())
    char_count = len(str(text))
    length_group = get_length_group(text)
    threshold = float(text_config.get("decision_threshold", 0.41)) if text_config else 0.41

    sentences = split_into_sentences(text)
    
    if len(sentences) >= 3 and word_count >= 50:
        chunks = create_sliding_windows(sentences, target_window_sentences=3, overlap=1)
    else:
        chunks = [str(text).strip()]

    all_texts_to_process = [str(text).strip()] + chunks
    X_all, ling_dicts = batch_extract_features(all_texts_to_process)

    probabilities = text_classifier.predict_proba(X_all)
    global_ai_prob = float(probabilities[0, 1])
    chunk_ai_probs = [float(probabilities[i, 1]) for i in range(1, len(all_texts_to_process))]

    agg_result = aggregate_document_predictions(
        global_ai_prob=global_ai_prob,
        chunk_probabilities=chunk_ai_probs,
        chunk_texts=chunks,
        threshold=threshold
    )

    doc_ling_dict = ling_dicts[0]
    human_prob = round((1.0 - global_ai_prob) * 100, 2)
    ai_prob = round(global_ai_prob * 100, 2)

    return {
        "success": True,
        "media_type": "text",
        "result": agg_result["result"],
        "is_short_text": False,
        "limited_context": False,
        "confidence": agg_result["confidence"],
        "ai_probability": ai_prob,
        "human_probability": human_prob,
        "word_count": word_count,
        "character_count": char_count,
        "length_group": length_group,
        "threshold": round(threshold, 2),
        "linguistic_features": doc_ling_dict,
        "global_ai_probability": agg_result["global_ai_probability"],
        "mean_chunk_ai_probability": agg_result["mean_chunk_ai_probability"],
        "peak_chunk_ai_probability": agg_result["peak_chunk_ai_probability"],
        "ai_chunk_ratio": agg_result["ai_chunk_ratio"],
        "number_of_chunks": agg_result["number_of_chunks"],
        "chunks": agg_result["chunks"]
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
        char_count = len(text)
        threshold = float(text_config.get("decision_threshold", 0.41)) if text_config else 0.41

        # --------------------------------------------------------
        # Tier 1: 1–2 Words (Insufficient Evidence)
        # --------------------------------------------------------
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
                "threshold": round(threshold, 2),
                "reason": "The input contains only 1–2 words. There is insufficient linguistic context for reliable AI detection."
            })

        # --------------------------------------------------------
        # Tier 2: 3–6 Words (Short Text Inference with Borderline Uncertainty)
        # --------------------------------------------------------
        if 3 <= word_count <= 6:
            X, ling_dicts = batch_extract_features([text])
            probabilities = text_classifier.predict_proba(X)[0]
            ai_prob_raw = float(probabilities[1])
            human_prob_raw = float(probabilities[0])

            ai_prob = round(ai_prob_raw * 100, 2)
            human_prob = round(human_prob_raw * 100, 2)

            # Borderline range [0.35, 0.50]
            if 0.35 <= ai_prob_raw <= 0.50:
                result = "UNCERTAIN"
                confidence = None
                reason = f"Short text ({word_count} words) with borderline linguistic signals ({ai_prob}% AI probability). Evidence is inconclusive."
            elif ai_prob_raw > 0.50:
                result = "AI-GENERATED"
                confidence = ai_prob
                reason = f"Short text ({word_count} words) exhibiting strong AI linguistic signals."
            else:
                result = "HUMAN-WRITTEN"
                confidence = human_prob
                reason = f"Short text ({word_count} words) exhibiting human linguistic signals."

            return jsonify({
                "success": True,
                "media_type": "text",
                "result": result,
                "is_short_text": True,
                "limited_context": True,
                "classification_method": "short_text_ml",
                "confidence": confidence,
                "ai_probability": ai_prob,
                "human_probability": human_prob,
                "word_count": word_count,
                "character_count": char_count,
                "length_group": "very_short",
                "threshold": round(threshold, 2),
                "reason": reason,
                "linguistic_features": ling_dicts[0]
            })

        # --------------------------------------------------------
        # Tier 3: 7+ Words (Normal & Long Multi-Chunk Inference)
        # --------------------------------------------------------
        result_payload = run_text_inference_pipeline(text)
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
