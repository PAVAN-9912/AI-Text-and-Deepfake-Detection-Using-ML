"""
Production Video Fusion Engine (Phase 8.2 — Restored Frozen Policy C)
====================================================================
Promotes frozen ST-SeqNet (Epoch 8) + frozen FaceViT + audited frozen Policy C
(Evidence-Gated Specialist Fusion) into production inference.

Strict Invariants:
- ST-SeqNet E8 checkpoint SHA-256 verified:
  1d68e7418f37fd59ed09619b3bbb7787dc26479336052c7e30472c8fa835a49c
- ST-SeqNet threshold: 0.50
- FaceViT checkpoint: experiments/v12_vit/checkpoints/vit_deepfake_best.pth
- FaceViT threshold: 0.475
- Policy C router: src/experimental/phase6_4_staging/staging_fusion_router.py (unchanged)
- Standard preprocessing: src/experimental/phase7f/preprocessing.py (unchanged)
"""

import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch

from src.experimental.phase7f.models import STSeqNet
from src.experimental.phase7f.preprocessing import StandardVideoPreprocessor
from src.predict_video_vit import predict_video_vit, load_vit_artifacts
from src.experimental.phase6_4_staging.staging_result_schema import (
    ExpertInferenceResult,
    ExpertStatus,
    FusionVerdict,
)
from src.experimental.phase6_4_staging.staging_fusion_router import StagingFusionRouter

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


def predict_video_fusion(video_path: str, mode: str = "decisive") -> Dict[str, Any]:
    """
    Runs frozen ST-SeqNet (Full-Frame Temporal, 16 frames) + frozen FaceViT
    (Localized Facial, 8 frames) and evaluates both signals through the frozen
    Policy C router (StagingFusionRouter.evaluate(..., policy="POLICY_C")).

    In mode="decisive" (default), Policy C INCONCLUSIVE domain conflicts are
    resolved into a decisive REAL (Human-Generated) or FAKE (AI-Generated)
    verdict using calibrated expert weighting. Pass mode="policy_c" for raw
    frozen Policy C abstention behavior.
    """
    if not _loaded:
        load_fusion_artifacts()

    t_total_start = time.time()

    # 1. ST-SeqNet Inference (Full-Frame Temporal, 16 frames via StandardVideoPreprocessor)
    t_st0 = time.time()
    st_p_ai: Optional[float] = None
    st_label = "ERROR"
    st_status = "UNKNOWN"
    st_err: Optional[str] = None
    attention_weights: List[float] = []

    try:
        frames, status = _preprocessor.decode_video(video_path)
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

    # Handle complete decode failure cleanly
    if st_status != "SUCCESS" and vit_status == "ERROR":
        return {
            "label": "ERROR",
            "verdict": "ERROR",
            "error": st_err or vit_err or "Failed to decode video.",
            "model": "PolicyC_STSeqNet_FaceViT",
        }

    # 3. Build ExpertInferenceResult payloads for frozen Policy C router
    if vit_status == "ERROR":
        exp_vit_stat = ExpertStatus.ERROR
        v_has_face = False
        v_prob = None
    elif vit_label == "INSUFFICIENT_EVIDENCE" or vit_faces == 0:
        exp_vit_stat = ExpertStatus.INSUFFICIENT_EVIDENCE
        v_has_face = False
        v_prob = None
    else:
        exp_vit_stat = ExpertStatus.SUCCESS
        v_has_face = True
        v_prob = vit_p_fake

    expert_vit = ExpertInferenceResult(
        expert_id="face_vit",
        expert_name="FaceViT_Base",
        status=exp_vit_stat,
        raw_prediction=1 if vit_label == "AI_GENERATED" else (0 if vit_label == "GENUINE" else -1),
        probability=v_prob,
        confidence=abs(v_prob - 0.5) * 2.0 if v_prob is not None else 0.0,
        face_detected=v_has_face,
        evidence_status=vit_label,
        execution_time_ms=vit_lat_ms,
        metadata={"face_count": vit_faces},
    )

    expert_st = ExpertInferenceResult(
        expert_id="st_seqnet",
        expert_name="STSeqNet_FullFrame",
        status=ExpertStatus.SUCCESS if st_status == "SUCCESS" else ExpertStatus.ERROR,
        raw_prediction=1 if st_label == "AI_GENERATED" else (0 if st_label == "GENUINE" else -1),
        probability=st_p_ai,
        confidence=abs(st_p_ai - 0.5) * 2.0 if st_p_ai is not None else 0.0,
        face_detected=False,
        evidence_status=st_label,
        execution_time_ms=st_lat_ms,
        metadata={"attention_weights": attention_weights},
    )

    # Direct invocation of the frozen Policy C router
    fus_res = StagingFusionRouter.evaluate(
        face_vit_result=expert_vit,
        exp_d_result=expert_st,
        policy="POLICY_C",
        total_latency_ms=st_lat_ms + vit_lat_ms,
        peak_ram_mb=0.0,
        input_video_path=video_path,
    )

    total_latency_ms = round((time.time() - t_total_start) * 1000.0, 1)

    routing_rule = fus_res.decision_rule
    explanation = fus_res.explanation
    is_abstained = bool(fus_res.is_abstained)

    if fus_res.verdict == FusionVerdict.AI_GENERATED:
        verdict = "FAKE"
        display_verdict = "AI-GENERATED"
        fake_prob = float(fus_res.calibrated_score) if fus_res.calibrated_score is not None else float(st_p_ai or 0.5)
        real_prob = 1.0 - fake_prob
    elif fus_res.verdict == FusionVerdict.LIKELY_HUMAN:
        verdict = "REAL"
        display_verdict = "HUMAN"
        fake_prob = float(fus_res.calibrated_score) if fus_res.calibrated_score is not None else float(st_p_ai or 0.5)
        real_prob = 1.0 - fake_prob
    elif mode == "decisive" and (st_p_ai is not None or vit_p_fake is not None):
        if st_p_ai is not None and vit_p_fake is not None:
            if vit_p_fake <= 0.20 and st_p_ai >= 0.80:
                fake_prob = 0.35 * st_p_ai + 0.65 * vit_p_fake
            else:
                fake_prob = 0.50 * st_p_ai + 0.50 * vit_p_fake
            verdict = "FAKE" if fake_prob >= 0.50 else "REAL"
        elif vit_p_fake is not None:
            fake_prob = float(vit_p_fake)
            verdict = "FAKE" if fake_prob >= VIT_THRESHOLD else "REAL"
        else:
            fake_prob = float(st_p_ai)
            verdict = "FAKE" if fake_prob >= ST_SEQNET_THRESHOLD else "REAL"
        real_prob = 1.0 - fake_prob
        display_verdict = "AI-GENERATED" if verdict == "FAKE" else "HUMAN"
        routing_rule = f"{fus_res.decision_rule}_RESOLVED_DECISIVE"
        is_abstained = False
        explanation = (
            f"Decisive resolution of {fus_res.decision_rule}: "
            f"FaceViT={(vit_p_fake or 0.0):.4f}, "
            f"ST-SeqNet={(st_p_ai or 0.0):.4f} -> {display_verdict}."
        )
    else:
        verdict = "INCONCLUSIVE"
        display_verdict = "INCONCLUSIVE"
        fake_prob = None
        real_prob = None

    active_threshold = ST_SEQNET_THRESHOLD if not v_has_face else VIT_THRESHOLD

    return {
        "label": verdict,
        "verdict": verdict,
        "display_verdict": display_verdict,
        "probabilities": {
            "fake": round(fake_prob, 4) if fake_prob is not None else None,
            "real": round(real_prob, 4) if real_prob is not None else None,
        },
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "st_seqnet_p_ai": round(st_p_ai, 6) if st_p_ai is not None else None,
        "vit_p_fake": round(vit_p_fake, 6) if vit_p_fake is not None else None,
        "face_count": vit_faces,
        "face_detected_frames": vit_faces,
        "frames_analyzed": 16,
        "vit_frames_analyzed": vit_frames_analyzed,
        "routing_rule": routing_rule,
        "domain_conflict": bool(fus_res.domain_conflict),
        "is_abstained": is_abstained,
        "reason": explanation,
        "attention_weights": attention_weights,
        "frame_probabilities": vit_frame_probs,
        "threshold": active_threshold,
        "model": "PolicyC_STSeqNet_FaceViT",
        "primary_expert": "PolicyC_STSeqNet_FaceViT",
        "routing_reason": explanation,
        "active_experts": ["STSeqNet_Epoch8", "FaceViT_Base"],
        "reliability": "INCONCLUSIVE" if verdict == "INCONCLUSIVE" else ("HIGH" if v_has_face else "FULL_FRAME"),
        "reliability_note": (
            "Domain Conflict resolved via decisive expert weighting."
            if (fus_res.domain_conflict and not is_abstained)
            else (
                "Domain Conflict: ST-SeqNet and FaceViT produced contradictory signals."
                if fus_res.domain_conflict
                else ("Faceless media routed via ST-SeqNet full-frame temporal analysis." if not v_has_face else "")
            )
        ),
        "processing_time_ms": total_latency_ms,
    }
