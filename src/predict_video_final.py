import os
import sys
import pickle
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "video_final"
FACE_DETECTOR_PATH = BASE_DIR / "models" / "face_detector" / "face_detection_yunet.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture definition
class FaceSpatialTemporalModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.25)
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.20
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        proj = self.projection(x)
        gru_out, _ = self.gru(proj)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        mean_pool = norm_out.mean(dim=1)
        max_pool = norm_out.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.classifier(pooled).squeeze(1)
        return logits

# Global cache for inference speed
_loaded = False
_model = None
_resnet = None
_transforms = None
_scaler_mean = None
_scaler_std = None
_calibrator = None
_threshold = 0.50
_face_detector = None

def load_video_artifacts():
    global _loaded, _model, _resnet, _transforms, _scaler_mean, _scaler_std, _calibrator, _threshold, _face_detector
    if _loaded:
        return

    # Load Face Detector
    if os.path.exists(FACE_DETECTOR_PATH):
        _face_detector = cv2.FaceDetectorYN_create(
            model=str(FACE_DETECTOR_PATH),
            config="",
            input_size=(224, 224),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=10
        )

    # Load Backbone
    weights = ResNet18_Weights.DEFAULT
    _resnet = resnet18(weights=weights)
    _resnet.fc = nn.Identity()
    _resnet.to(DEVICE)
    _resnet.eval()
    _transforms = weights.transforms()

    # Load Scaler
    scaler_path = MODEL_DIR / "video_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        _scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
        _scaler_std = np.asarray(scaler["std"], dtype=np.float32)

    # Load Model Weights
    model_path = MODEL_DIR / "temporal_attention_model.pth"
    _model = FaceSpatialTemporalModel(input_dim=512, hidden_dim=128)
    _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()

    # Load Calibrator & Threshold
    cal_path = MODEL_DIR / "probability_calibrator.pkl"
    with open(cal_path, "rb") as f:
        cal_data = pickle.load(f)
        _calibrator = cal_data["model"]
        _threshold = float(cal_data.get("threshold", 0.50))

    _loaded = True

def extract_face_crop(frame, target_size=(224, 224), margin=0.25):
    h, w, _ = frame.shape
    if _face_detector is not None:
        _face_detector.setInputSize((w, h))
        _, faces = _face_detector.detect(frame)
        if faces is not None and len(faces) > 0:
            best_face = max(faces, key=lambda f: f[-1])
            x, y, fw, fh = best_face[0:4].astype(int)
            mx, my = int(fw * margin), int(fh * margin)
            x1, y1 = max(0, x - mx), max(0, y - my)
            x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA), True
    min_dim = min(h, w)
    cy, cx = h // 2, w // 2
    crop = frame[max(0, cy - min_dim // 2):max(0, cy - min_dim // 2) + min_dim, max(0, cx - min_dim // 2):max(0, cx - min_dim // 2) + min_dim]
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA), False

def predict_video(video_path):
    load_video_artifacts()
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    if total_frames >= 8:
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (8 - total_frames)

    tensor_list = []
    face_detected_count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        crop, detected = extract_face_crop(frame)
        if detected:
            face_detected_count += 1
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = _transforms(torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0)
        tensor_list.append(tensor)

    cap.release()

    batch_tensors = torch.stack(tensor_list).to(DEVICE)
    with torch.no_grad():
        feats = _resnet(batch_tensors).cpu().numpy().astype(np.float32) # (8, 512)

    # Standardize
    feats_std = (feats - _scaler_mean) / _scaler_std
    input_tensor = torch.tensor(feats_std, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, 8, 512)

    with torch.no_grad():
        logit = _model(input_tensor).cpu().numpy().reshape(-1, 1)

    # Platt calibrated probabilities: 0 = Fake, 1 = Real
    cal_probs = _calibrator.predict_proba(logit)[0]
    real_prob = float(cal_probs[1])
    fake_prob = float(cal_probs[0])

    is_real = real_prob >= _threshold
    label = "Real / Authentic" if is_real else "AI Generated / Deepfake"
    confidence = real_prob if is_real else fake_prob

    return {
        "label": label,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "confidence": round(confidence, 4),
        "threshold": round(_threshold, 2),
        "frames_analyzed": len(tensor_list),
        "face_detected_frames": face_detected_count,
        "model": "FaceSpatialTemporalModel_V3"
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    else:
        test_video = str(BASE_DIR / "datasets" / "video" / "prepared" / "test" / "real" / "00000.mp4")

    res = predict_video(test_video)
    print("=" * 60)
    print("FINAL VIDEO DEEPFAKE DETECTOR INFERENCE")
    print("=" * 60)
    print(f"Video File           : {test_video}")
    print(f"Label                : {res['label']}")
    print(f"Deepfake Probability : {res['fake_probability']*100:.2f}%")
    print(f"Real Probability     : {res['real_probability']*100:.2f}%")
    print(f"Confidence           : {res['confidence']*100:.2f}%")
    print(f"Decision Threshold   : {res['threshold']:.2f}")
    print(f"Face Detected Frames : {res['face_detected_frames']}/{res['frames_analyzed']}")
    print("=" * 60)
