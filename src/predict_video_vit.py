"""
ViT Inference Module: predict_video_vit.py
==========================================
Isolated inference module for the Vision Transformer (ViT) deepfake detector.

Architecture:
    Backbone: google/vit-base-patch16-224
    Classification Head: 2 classes (0: FAKE, 1: REAL)
    Input Size: 224x224
    Normalization: Mean [0.5, 0.5, 0.5], Std [0.5, 0.5, 0.5]
    Channel Order: RGB (converted from OpenCV BGR)

Decision Rule:
    REAL if P(Real) >= THRESHOLD (calibrated)
    FAKE otherwise
    INSUFFICIENT_EVIDENCE if no face detected in video
"""

import os
import io
import sys
import json
import logging
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import ViTForImageClassification

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
VIT_CHECKPOINT = (
    BASE_DIR
    / "experiments"
    / "v12_vit"
    / "checkpoints"
    / "vit_deepfake_best.pth"
)
FACE_DETECTOR_PATH = BASE_DIR / "models" / "face_detector" / "face_detection_yunet.onnx"

# ── Constants ─────────────────────────────────────────────────────────────────
VIT_MEAN         = [0.5, 0.5, 0.5]
VIT_STD          = [0.5, 0.5, 0.5]
INPUT_SIZE       = 224
N_FRAMES         = 8        # uniform temporal samples per video
N_CROPS_MAX      = 8        # max face crops to aggregate
FACE_SCORE_THR   = 0.55
FACE_MARGIN      = 0.25
DEFAULT_THRESHOLD = 0.50

# ── Cached singletons ─────────────────────────────────────────────────────────
_model: Optional[nn.Module] = None
_face_detector = None
_FACE_DETECTOR_LOCK = threading.Lock()
_loaded: bool = False
_threshold: float = DEFAULT_THRESHOLD

DEVICE = torch.device("cpu")

# ── Preprocessing Transform ───────────────────────────────────────────────────
_inference_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


def load_vit_artifacts(checkpoint_path: Optional[Path] = None) -> None:
    """Load ViT model, calibration threshold, and face detector into module-level cache."""
    global _model, _face_detector, _loaded, _threshold

    if _loaded:
        return

    target_ckpt = checkpoint_path or VIT_CHECKPOINT

    if not target_ckpt.exists():
        raise FileNotFoundError(
            f"ViT checkpoint not found: {target_ckpt}\n"
            "Run experiments/v12_vit/train_vit.py first."
        )

    # Verify SHA-256 if metadata file exists
    meta_path = target_ckpt.parent / "vit_best_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_sha = meta.get("sha256", "")
            if expected_sha:
                actual_sha = hashlib.sha256(target_ckpt.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    logger.warning(f"ViT checkpoint SHA-256 mismatch (Expected {expected_sha}, got {actual_sha})")
                else:
                    logger.info(f"ViT checkpoint SHA-256 verified: {actual_sha[:16]}...")
        except Exception as e:
            logger.warning(f"Could not verify checkpoint metadata: {e}")

    # Initialize ViT architecture (prefer local cache for fast offline startup)
    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            id2label={0: "FAKE", 1: "REAL"},
            label2id={"FAKE": 0, "REAL": 1},
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    except Exception:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            id2label={0: "FAKE", 1: "REAL"},
            label2id={"FAKE": 0, "REAL": 1},
            ignore_mismatched_sizes=True,
        )

    state = torch.load(str(target_ckpt), map_location=DEVICE)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    _model = model

    val_auc = state.get("val_auc", float("nan"))
    logger.info(f"ViT deepfake detector loaded successfully (Val AUC: {val_auc})")

    # Load calibrated threshold
    cal_path = target_ckpt.parent / "vit_threshold.json"
    if cal_path.exists():
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            _threshold = float(cal.get("threshold", DEFAULT_THRESHOLD))
            logger.info(f"ViT calibrated threshold loaded: {_threshold:.4f}")
        except Exception:
            _threshold = DEFAULT_THRESHOLD
    else:
        _threshold = DEFAULT_THRESHOLD
        logger.info(f"ViT using default threshold: {_threshold:.4f}")

    # Load Face Detector
    if FACE_DETECTOR_PATH.exists():
        _face_detector = cv2.FaceDetectorYN_create(
            model=str(FACE_DETECTOR_PATH),
            config="",
            input_size=(INPUT_SIZE, INPUT_SIZE),
            score_threshold=FACE_SCORE_THR,
            nms_threshold=0.3,
            top_k=10,
        )
        logger.info("YuNet face detector loaded for ViT")
    else:
        _face_detector = None
        logger.warning(f"YuNet face detector not found at {FACE_DETECTOR_PATH}")

    _loaded = True


def _extract_face_crop_vit(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract face crop from a single frame using YuNet.
    Returns (crop_bgr, had_face).
    """
    if _face_detector is None:
        return None, False

    h, w = frame_bgr.shape[:2]
    with _FACE_DETECTOR_LOCK:
        _face_detector.setInputSize((w, h))
        _, faces = _face_detector.detect(frame_bgr)

    if faces is None or len(faces) == 0:
        return None, False

    # Select highest-confidence detection
    best = max(faces, key=lambda f: float(f[14]))
    x, y, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])

    mx = int(bw * FACE_MARGIN)
    my = int(bh * FACE_MARGIN)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, False

    crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return crop, True


def predict_video_vit(video_path: str) -> Dict[str, Any]:
    """
    Run Vision Transformer (ViT) inference on a video file.

    Returns a dict with:
      - label: "REAL" | "FAKE" | "INSUFFICIENT_EVIDENCE"
      - real_probability: float [0, 1]
      - fake_probability: float [0, 1]
      - confidence: float [0, 1]
      - face_count: int
      - reliability: "HIGH" | "MODERATE" | "LOW" | "NONE"
      - model_version: "v12_vit"
      - model: "VideoDetector_ViT_Base"
      - threshold: float
      - frame_probs: List[float] (per-crop real probabilities)
    """
    if not _loaded:
        load_vit_artifacts()

    vpath = str(video_path)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return _error_response(f"Cannot open video: {vpath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return _error_response("Video has zero frames")

    # Uniform temporal frame sampling
    total_safe = max(1, total_frames - 5)
    sample_indices = sorted(set(
        int(total_safe * i / max(1, N_FRAMES - 1))
        for i in range(N_FRAMES)
    ))
    sample_indices = [min(i, total_safe - 1) for i in sample_indices]

    face_crops_rgb = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        crop, had_face = _extract_face_crop_vit(frame)
        if had_face and crop is not None:
            # OpenCV BGR -> RGB conversion
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_crops_rgb.append(crop_rgb)
        if len(face_crops_rgb) >= N_CROPS_MAX:
            break

    cap.release()

    face_count = len(face_crops_rgb)
    if face_count == 0:
        return _insufficient_evidence_response(len(sample_indices))

    # Run ViT inference
    tensors = [_inference_transform(c) for c in face_crops_rgb]
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        outputs = _model(batch)
        logits = outputs.logits  # (M, 2)
        probs = torch.softmax(logits, dim=-1)
        real_probs = probs[:, 1].cpu().tolist()

    real_prob = float(np.mean(real_probs))
    fake_prob = 1.0 - real_prob

    # Decision rule: REAL if real_prob >= threshold; otherwise FAKE
    if real_prob >= _threshold:
        label = "REAL"
    else:
        label = "FAKE"

    confidence = abs(real_prob - 0.5) * 2.0

    if face_count >= 4:
        reliability = "HIGH"
        rel_note = "Multi-frame face sequence extracted; ViT patch self-attention verified."
    elif face_count >= 2:
        reliability = "MODERATE"
        rel_note = "Moderate face sequence extracted; prediction is reliable."
    else:
        reliability = "LOW"
        rel_note = "Only 1 valid face crop extracted; limited multi-frame evidence."

    return {
        "label": label,
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "confidence": round(confidence, 4),
        "face_count": face_count,
        "reliability": reliability,
        "reliability_note": rel_note,
        "model_version": "v12_vit",
        "model": "VideoDetector_ViT_Base",
        "threshold": _threshold,
        "frame_probs": [round(p, 4) for p in real_probs],
        "frame_probabilities": [round(p, 4) for p in real_probs],
        "frames_analyzed": len(sample_indices),
        "face_detected_frames": face_count,
        "sampled_frames": len(sample_indices),
        "decoded_frames": len(sample_indices),
        "detected_faces": face_count,
        "accepted_crops": face_count,
        "rejected_frames": max(0, len(sample_indices) - face_count),
        "reason": "Analyzed using Vision Transformer (ViT-Base patch16-224) multi-head self-attention detector."
    }


def _error_response(msg: str) -> Dict[str, Any]:
    logger.error(f"ViT inference error: {msg}")
    return {
        "label": "ERROR",
        "real_probability": 0.0,
        "fake_probability": 0.0,
        "confidence": 0.0,
        "face_count": 0,
        "reliability": "NONE",
        "reliability_note": msg,
        "model_version": "v12_vit",
        "model": "VideoDetector_ViT_Base",
        "threshold": _threshold,
        "frame_probs": [],
        "frame_probabilities": [],
        "frames_analyzed": 0,
        "face_detected_frames": 0,
        "sampled_frames": 0,
        "decoded_frames": 0,
        "detected_faces": 0,
        "accepted_crops": 0,
        "rejected_frames": 0,
        "reason": f"Inference error: {msg}"
    }


def _insufficient_evidence_response(frames_analyzed: int = 0) -> Dict[str, Any]:
    return {
        "label": "INSUFFICIENT_EVIDENCE",
        "real_probability": 0.0,
        "fake_probability": 0.0,
        "confidence": 0.0,
        "face_count": 0,
        "reliability": "NONE",
        "reliability_note": "No human faces detected in any sampled frame. Cannot assess facial manipulation without face crops.",
        "model_version": "v12_vit",
        "model": "VideoDetector_ViT_Base",
        "threshold": _threshold,
        "frame_probs": [],
        "frame_probabilities": [],
        "frames_analyzed": frames_analyzed,
        "face_detected_frames": 0,
        "sampled_frames": frames_analyzed,
        "decoded_frames": frames_analyzed,
        "detected_faces": 0,
        "accepted_crops": 0,
        "rejected_frames": frames_analyzed,
        "reason": "Zero face detections; facial manipulation classifier requires valid face crops."
    }
