"""
Production Video Fusion Engine (Phase 8)
======================================
Promotes frozen ST-SeqNet (Epoch 8) + frozen FaceViT + audited Policy C
(Evidence-Gated Specialist Fusion) into production inference.

Strict Invariants:
- ST-SeqNet E8 checkpoint SHA-256 verified:
  1d68e7418f37fd59ed09619b3bbb7787dc26479336052c7e30472c8fa835a49c
- ST-SeqNet threshold: 0.50
- FaceViT checkpoint: experiments/v12_vit/checkpoints/vit_deepfake_best.pth
- FaceViT threshold: 0.475
- Policy C router: src/experimental/phase6_4_staging/staging_fusion_router.py (unchanged)
"""

import os
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import torch

from src.experimental.phase7f.models import STSeqNet
from src.experimental.phase7f.preprocessing import StandardVideoPreprocessor
from src.predict_video_vit import predict_video_vit, load_vit_artifacts

BASE_DIR = Path(__file__).resolve().parent.parent

ST_SEQNET_CHECKPOINT_PATH = (
    BASE_DIR / "experiments" / "phase7f_4" / "checkpoints" / "st_seqnet_best_checkpoint.pth"
)
EXPECTED_ST_SEQNET_SHA256 = (
    "1d68e7418f37fd59ed09619b3bbb7787dc26479336052c7e30472c8fa835a49c"
)
ST_SEQNET_THRESHOLD = 0.50

VIT_CHECKPOINT_PATH = (
    BASE_DIR / "experiments" / "v12_vit" / "checkpoints" / "vit_deepfake_best.pth"
)
VIT_THRESHOLD = 0.475

_fusion_lock = threading.Lock()
_st_model: Optional[STSeqNet] = None
_preprocessor: Optional[StandardVideoPreprocessor] = None
_loaded: bool = False


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def load_fusion_artifacts(
    st_checkpoint_path: Optional[Path] = None,
    vit_checkpoint_path: Optional[Path] = None,
) -> None:
    """Thread-safe eager initialization of ST-SeqNet (E8) and FaceViT artifacts."""
    global _st_model, _preprocessor, _loaded

    with _fusion_lock:
        if _loaded:
            return

        st_ckpt = st_checkpoint_path or ST_SEQNET_CHECKPOINT_PATH
        vit_ckpt = vit_checkpoint_path or VIT_CHECKPOINT_PATH

        if not st_ckpt.exists():
            raise FileNotFoundError(f"ST-SeqNet checkpoint not found: {st_ckpt}")

        actual_sha = _compute_sha256(st_ckpt)
        if actual_sha != EXPECTED_ST_SEQNET_SHA256:
            raise ValueError(
                f"ST-SeqNet checkpoint SHA-256 mismatch! Expected {EXPECTED_ST_SEQNET_SHA256}, got {actual_sha}"
            )

        # Load ST-SeqNet on CPU/CUDA cleanly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = STSeqNet(pretrained_backbone=False)
        ckpt_data = torch.load(st_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        _st_model = model
        _preprocessor = StandardVideoPreprocessor()

        # Load FaceViT artifacts (verifies its own SHA-256 and calibrated threshold 0.475)
        load_vit_artifacts(checkpoint_path=vit_ckpt)

        _loaded = True


def _decode_frames_with_border_crop(
    video_path: str, preprocessor: StandardVideoPreprocessor
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Decodes 16 uniformly sampled frames via StandardVideoPreprocessor sampling indices,
    stripping solid black pillarbox/letterbox bars before 224x224 resizing when
    padding exceeds 15% of frame area (so vertical 9:16 Reels/Shorts are not diluted).
    """
    status: Dict[str, Any] = {
        "success": False,
        "total_frames_reported": 0,
        "frames_decoded": 0,
        "border_cropped": False,
        "error_msg": None,
    }
    if not os.path.exists(video_path):
        status["error_msg"] = f"File not found: {video_path}"
        return None, status

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status["error_msg"] = f"Failed to open video stream: {video_path}"
        return None, status

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        count = 0
        while cap.grab():
            count += 1
            if count > 5000:
                break
        total_frames = count
    status["total_frames_reported"] = total_frames
    if total_frames == 0:
        cap.release()
        status["error_msg"] = "Zero frames readable from video stream"
        return None, status

    target_indices = preprocessor.sample_frame_indices(total_frames)
    selected_raw: List[Optional[np.ndarray]] = []
    last_valid = None
    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            selected_raw.append(frame)
            last_valid = frame
        elif last_valid is not None:
            selected_raw.append(last_valid)
        else:
            selected_raw.append(None)
    cap.release()

    first_valid = next((f for f in selected_raw if f is not None), None)
    if first_valid is None:
        status["error_msg"] = "All decoded target frames were null"
        return None, status
    valid_frames = [f if f is not None else first_valid for f in selected_raw]

    # Detect solid black pillarbox/letterbox bars (> 15% of frame area)
    h0, w0 = valid_frames[0].shape[:2]
    crop_box: Optional[Tuple[int, int, int, int]] = None
    if h0 >= 64 and w0 >= 64:
        sample_stack = np.max(np.stack(valid_frames[::4], axis=0), axis=(0, 3))  # [H, W]
        non_black = sample_stack >= 18
        rows = np.any(non_black, axis=1)
        cols = np.any(non_black, axis=0)
        if np.any(rows) and np.any(cols):
            r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1]) + 1
            c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1]) + 1
            cropped_area = (r_max - r_min) * (c_max - c_min)
            orig_area = h0 * w0
            if (cropped_area / float(orig_area)) <= 0.85 and (r_max - r_min) >= 64 and (c_max - c_min) >= 64:
                crop_box = (r_min, r_max, c_min, c_max)
                status["border_cropped"] = True

    processed = []
    for frame in valid_frames:
        if crop_box is not None:
            r_min, r_max, c_min, c_max = crop_box
            frame = frame[r_min:r_max, c_min:c_max]
        resized = cv2.resize(
            frame,
            (preprocessor.config.target_width, preprocessor.config.target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        processed.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    status["frames_decoded"] = len(processed)
    status["success"] = True
    return np.stack(processed, axis=0), status


def fuse_binary_probabilities(
    st_seqnet_p_ai: float,
    vit_p_fake: Optional[float],
    face_count: int,
) -> Tuple[float, str, str]:
    """
    Deterministic Soft Binary Fusion (threshold = 0.50, 0% abstention):
    - Case A (face_count == 0 or vit_p_fake is None): p_ai = st_seqnet_p_ai
    - Case B (face_count > 0 and vit_p_fake >= 0.55): p_ai = max(vit_p_fake, 0.5 * st + 0.5 * vit)
    - Case C (face_count > 0 and vit_p_fake <= 0.25 and st >= 0.70): p_ai = 0.35 * st + 0.65 * vit
    - Case D (all other face_count > 0): p_ai = 0.65 * st + 0.35 * vit
    """
    st_p = float(np.clip(st_seqnet_p_ai, 0.0, 1.0))
    if face_count <= 0 or vit_p_fake is None:
        return (
            st_p,
            "CASE_A_FACELESS_ST_SEQNET",
            f"Case A (Faceless): 100% ST-SeqNet full-frame temporal routing (P_ai={st_p:.4f}).",
        )

    vit_p = float(np.clip(vit_p_fake, 0.0, 1.0))
    if vit_p >= 0.55:
        p_ai = max(vit_p, 0.5 * st_p + 0.5 * vit_p)
        return (
            float(np.clip(p_ai, 0.0, 1.0)),
            "CASE_B_STRONG_FACIAL_MANIPULATION",
            f"Case B (Strong Facial Manipulation): FaceViT={vit_p:.4f} >= 0.55, ST-SeqNet={st_p:.4f} -> P_ai={p_ai:.4f}.",
        )

    if vit_p <= 0.25 and st_p >= 0.70:
        p_ai = 0.35 * st_p + 0.65 * vit_p
        return (
            float(np.clip(p_ai, 0.0, 1.0)),
            "CASE_C_AUTHENTIC_FACE_DAMPENING",
            f"Case C (Authentic Face Dampening): FaceViT={vit_p:.4f} <= 0.25 dampens ST-SeqNet={st_p:.4f} -> P_ai={p_ai:.4f}.",
        )

    p_ai = 0.65 * st_p + 0.35 * vit_p
    return (
        float(np.clip(p_ai, 0.0, 1.0)),
        "CASE_D_DUAL_EXPERT_BLEND",
        f"Case D (Dual-Expert Blend): 0.65*ST-SeqNet({st_p:.4f}) + 0.35*FaceViT({vit_p:.4f}) -> P_ai={p_ai:.4f}.",
    )


def predict_video_fusion(video_path: str) -> Dict[str, Any]:
    """
    Runs ST-SeqNet (Full-Frame Temporal) + FaceViT (Localized Facial) and applies
    deterministic Soft Binary Fusion (threshold=0.50), always returning FAKE or REAL.
    """
    if not _loaded:
        load_fusion_artifacts()

    t_total_start = time.time()

    # 1. ST-SeqNet Inference (Full-Frame Temporal, 16 frames with non-zero border crop)
    t_st0 = time.time()
    st_p_ai: Optional[float] = None
    st_label = "ERROR"
    st_status = "UNKNOWN"
    st_err: Optional[str] = None
    border_cropped = False
    attention_weights: List[float] = []

    try:
        frames, status = _decode_frames_with_border_crop(video_path, _preprocessor)
        border_cropped = bool(status.get("border_cropped", False))
        if not status["success"] or frames is None:
            st_status = "DECODING_ERROR"
            st_err = status.get("error_msg", "Decoding failed")
        else:
            device = next(_st_model.parameters()).device
            tensor = _preprocessor.frames_to_tensor(frames).unsqueeze(0).to(device)
            with torch.no_grad():
                out = _st_model(tensor)
                st_p_ai = float(out["probabilities"].item())
                attn = out.get("attention_weights")
                if attn is not None:
                    attention_weights = [round(float(w), 6) for w in attn.squeeze(0).tolist()]
            st_label = "AI_GENERATED" if st_p_ai >= ST_SEQNET_THRESHOLD else "GENUINE"
            st_status = "SUCCESS"
    except Exception as e:
        st_status = "ERROR"
        st_err = str(e)
    st_lat_ms = (time.time() - t_st0) * 1000.0

    # 2. FaceViT Inference (Localized Facial, 8 frames)
    t_vit0 = time.time()
    vit_p_fake: Optional[float] = None
    vit_label = "UNKNOWN"
    vit_faces = 0
    vit_status = "UNKNOWN"
    vit_err: Optional[str] = None
    vit_frames_analyzed = 8
    vit_frame_probs: List[float] = []
    vit_raw: Dict[str, Any] = {}

    try:
        vit_raw = predict_video_vit(video_path)
        vit_faces = int(vit_raw.get("face_detected_frames", vit_raw.get("face_count", 0)))
        vit_frames_analyzed = int(vit_raw.get("frames_analyzed", 8))
        vit_frame_probs = vit_raw.get("frame_probabilities", []) or []
        raw_lbl = vit_raw.get("label", "ERROR")
        vit_status = "SUCCESS"

        if raw_lbl == "INSUFFICIENT_EVIDENCE":
            vit_label = "INSUFFICIENT_EVIDENCE"
        elif raw_lbl in ("REAL", "FAKE"):
            vit_p_real = float(vit_raw.get("real_probability", 0.0))
            vit_p_fake = float(vit_raw.get("fake_probability", 1.0 - vit_p_real))
            vit_label = "AI_GENERATED" if raw_lbl == "FAKE" else "GENUINE"
        else:
            vit_label = "ERROR"
            vit_status = "ERROR"
            vit_err = vit_raw.get("error", "FaceViT inference error")
    except Exception as e:
        vit_status = "ERROR"
        vit_label = "ERROR"
        vit_err = str(e)
    vit_lat_ms = (time.time() - t_vit0) * 1000.0

    # Handle complete decode failure cleanly (or ST-SeqNet failure in decisive mode)
    if st_status != "SUCCESS" or st_p_ai is None:
        return {
            "label": "ERROR",
            "verdict": "ERROR",
            "error": st_err or vit_err or "Failed to decode video.",
            "model": "SoftBinaryFusion_STSeqNet_FaceViT",
        }

    v_has_face = vit_status == "SUCCESS" and vit_label != "INSUFFICIENT_EVIDENCE" and vit_faces > 0
    effective_face_count = vit_faces if v_has_face else 0

    # 3. Deterministic Soft Binary Fusion (Cases A-D, threshold=0.50)
    p_ai, routing_rule, reason = fuse_binary_probabilities(
        st_seqnet_p_ai=st_p_ai,
        vit_p_fake=vit_p_fake if v_has_face else None,
        face_count=effective_face_count,
    )

    verdict = "FAKE" if p_ai >= ST_SEQNET_THRESHOLD else "REAL"
    display_verdict = "AI-GENERATED" if verdict == "FAKE" else "HUMAN"
    fake_prob = p_ai
    real_prob = 1.0 - fake_prob
    total_latency_ms = round((time.time() - t_total_start) * 1000.0, 1)

    return {
        "label": verdict,
        "verdict": verdict,
        "display_verdict": display_verdict,
        "probabilities": {
            "fake": round(fake_prob, 4),
            "real": round(real_prob, 4),
        },
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "st_seqnet_p_ai": round(st_p_ai, 6),
        "vit_p_fake": round(vit_p_fake, 6) if vit_p_fake is not None else None,
        "face_count": effective_face_count,
        "face_detected_frames": effective_face_count,
        "frames_analyzed": 16,
        "vit_frames_analyzed": vit_frames_analyzed,
        "routing_rule": routing_rule,
        "domain_conflict": False,
        "is_abstained": False,
        "border_cropped": border_cropped,
        "reason": reason,
        "attention_weights": attention_weights,
        "frame_probabilities": vit_frame_probs,
        "threshold": ST_SEQNET_THRESHOLD,
        "model": "PolicyC_STSeqNet_FaceViT",
        "primary_expert": "SoftBinaryFusion_STSeqNet_FaceViT",
        "routing_reason": reason,
        "active_experts": ["STSeqNet_Epoch8", "FaceViT_Base"],
        "reliability": "HIGH" if v_has_face else "FULL_FRAME",
        "reliability_note": (
            "Faceless media routed 100% via ST-SeqNet full-frame temporal analysis."
            if not v_has_face else ""
        ),
        "processing_time_ms": total_latency_ms,
    }
