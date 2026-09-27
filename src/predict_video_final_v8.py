import os
import sys
import pickle
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "video_final_v8"
FACE_DETECTOR_PATH = BASE_DIR / "models" / "face_detector" / "face_detection_yunet.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture Definition matching VideoDetector_V8 (QualityGatedTemporalModel)
class QualityGatedTemporalModel(nn.Module):
    def __init__(self, input_dim=512, quality_dim=3, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.GELU()
        self.q_gate = nn.Sequential(
            nn.Linear(quality_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
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

    def forward(self, x, q):
        x_t = x.transpose(1, 2)
        c_out = self.act1(self.bn1(self.conv1(x_t))).transpose(1, 2)
        gate = self.q_gate(q)
        gated_c = c_out * gate
        gru_out, _ = self.gru(gated_c)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Global model cache
_loaded = False
_model = None
_resnet = None
_transforms = None
_scaler_mean = None
_scaler_std = None
_calibrator = None
_threshold = 0.19
_face_detector = None

def load_video_artifacts():
    global _loaded, _model, _resnet, _transforms, _scaler_mean, _scaler_std, _calibrator, _threshold, _face_detector
    if _loaded:
        return

    # 1. Face Detector (YuNet)
    if os.path.exists(FACE_DETECTOR_PATH):
        _face_detector = cv2.FaceDetectorYN_create(
            model=str(FACE_DETECTOR_PATH),
            config="",
            input_size=(224, 224),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=10
        )

    # 2. ResNet-18 Backbone
    weights = ResNet18_Weights.DEFAULT
    _resnet = resnet18(weights=weights)
    _resnet.fc = nn.Identity()
    _resnet.to(DEVICE)
    _resnet.eval()
    _transforms = weights.transforms()

    # 3. Scaler
    scaler_path = MODEL_DIR / "video_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        _scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
        _scaler_std = np.asarray(scaler["std"], dtype=np.float32)

    # 4. Quality-Gated Model
    model_path = MODEL_DIR / "temporal_attention_model.pth"
    _model = QualityGatedTemporalModel(input_dim=512, quality_dim=3, hidden_dim=128)
    _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _model.to(DEVICE)
    _model.eval()

    # 5. Calibrator & Threshold
    cal_path = MODEL_DIR / "probability_calibrator.pkl"
    with open(cal_path, "rb") as f:
        cal_data = pickle.load(f)
        _calibrator = cal_data["model"] if isinstance(cal_data, dict) else cal_data
        _threshold = float(cal_data.get("threshold", 0.19))

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

    target_count = 8
    frame_indices = np.linspace(0, max(0, total_frames - 1), target_count, dtype=int)
    frames = []
    face_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        crop, has_face = extract_face_crop(frame)
        if has_face:
            face_count += 1
        frames.append(crop)

    cap.release()

    while len(frames) < target_count:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    frames = frames[:target_count]

    # Extract spatial ResNet-18 features
    feature_list = []
    with torch.no_grad():
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_img = torch.tensor(rgb_frame, dtype=torch.uint8).permute(2, 0, 1)
            tensor_transformed = _transforms(tensor_img).unsqueeze(0).to(DEVICE)
            feat = _resnet(tensor_transformed).cpu().numpy().flatten()
            feature_list.append(feat)

    features = np.stack(feature_list, axis=0) # [8, 512]
    features_norm = (features - _scaler_mean) / _scaler_std # [8, 512]

    # Compute Quality Metadata Descriptors
    norms = np.linalg.norm(features_norm, axis=1, keepdims=True) # [8, 1]
    unit_feats = features_norm / (norms + 1e-8)
    cos_sim = np.ones((8, 1), dtype=np.float32)
    for t in range(1, 8):
        cos_sim[t, 0] = np.sum(unit_feats[t] * unit_feats[t-1])
    face_ratio = np.full((8, 1), fill_value=face_count / 8.0, dtype=np.float32)
    
    q_feats = np.concatenate([norms / 25.0, cos_sim, face_ratio], axis=1) # [8, 3]

    feat_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, 8, 512]
    q_tensor = torch.tensor(q_feats, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, 8, 3]

    with torch.no_grad():
        logit = _model(feat_tensor, q_tensor).cpu().numpy().reshape(-1, 1)

    cal_probs = _calibrator.predict_proba(logit)[0]
    p_fake = float(cal_probs[0])
    p_real = float(cal_probs[1])

    prob_sum = p_fake + p_real
    p_fake = p_fake / prob_sum
    p_real = 1.0 - p_fake

    is_real = p_real >= _threshold
    label = "Real Video" if is_real else "AI Generated / Deepfake"
    confidence = p_real if is_real else p_fake

    return {
        "label": label,
        "fake_probability": round(p_fake, 4),
        "real_probability": round(p_real, 4),
        "confidence": round(confidence, 4),
        "threshold": round(_threshold, 4),
        "frames_analyzed": target_count,
        "face_detected_frames": face_count,
        "model": "VideoDetector_V8"
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    else:
        test_video = r"C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection\datasets\video\celebdf_raw\Celeb-synthesis\id0_id16_0003.mp4"

    res = predict_video(test_video)
    print("=" * 60)
    print("VIDEO DETECTOR V8 QUALITY-GATED INFERENCE")
    print("=" * 60)
    print(f"Video Path       : {test_video}")
    print(f"Label            : {res['label']}")
    print(f"Deepfake Prob    : {res['fake_probability']*100:.2f}%")
    print(f"Real Prob        : {res['real_probability']*100:.2f}%")
    print(f"Confidence       : {res['confidence']*100:.2f}%")
    print(f"Threshold        : {res['threshold']:.4f}")
    print(f"Faces Detected   : {res['face_detected_frames']} / {res['frames_analyzed']}")
    print(f"Model            : {res['model']}")
    print("=" * 60)
