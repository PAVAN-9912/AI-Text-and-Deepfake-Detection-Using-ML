"""
Staging Video Detection Engine
==============================
Main staging orchestrator for Phase 6.4 Two-Expert Video Detection.

Executes:
  Expert 1: Production Face ViT (via untouched predict_video_vit)
  Expert 2: Phase 6.2B Experiment D ConvNeXt-Wavelet (via frozen checkpoint)
  Router:   Policy C / Policy D StagingFusionRouter

Guarantees:
  - Checkpoint integrity pre-verification.
  - Complete isolation from production user-facing endpoints.
  - Bounded concurrency and resource safety via ResourceManager.
  - Results strictly tagged as EXPERIMENTAL_STAGING_RESULTS.
"""

import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import logging

from .staging_result_schema import (
    ExpertInferenceResult,
    FusionDecisionResult,
    FusionVerdict,
    ExpertStatus
)
from .staging_policy_config import (
    VIT_CHECKPOINT_PATH,
    VIT_CHECKPOINT_SHA256,
    EXPD_CHECKPOINT_PATH,
    EXPD_CHECKPOINT_SHA256,
    DEFAULT_INFERENCE_TIMEOUT_SEC
)
from .staging_resource_manager import ResourceManager
from .staging_video_preprocessor import extract_uniform_frames, VideoPreprocessingError
from .staging_fusion_router import StagingFusionRouter

# Import untouched production detector
from src.predict_video_vit import predict_video_vit
# Import ConvNeXt-Wavelet architecture
from src.experimental.phase6_2a_wavelet.convnext_wavelet import ConvNeXtWaveletDetector

logger = logging.getLogger("staging_video_engine")


class StagingVideoEngine:
    """Orchestrates two-expert video inference and fusion in staging."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.resource_mgr = ResourceManager()
        self._exp_d_model: Optional[ConvNeXtWaveletDetector] = None
        self._integrity_verified: bool = False

    def verify_checkpoints(self) -> Dict[str, Any]:
        """Cryptographically verifies that neither checkpoint has been modified."""
        results = {}
        for name, path, expected_hash in [
            ("production_face_vit", VIT_CHECKPOINT_PATH, VIT_CHECKPOINT_SHA256),
            ("exp_d_wavelet", EXPD_CHECKPOINT_PATH, EXPD_CHECKPOINT_SHA256)
        ]:
            if not path.exists():
                results[name] = {"status": "MISSING", "path": str(path)}
                continue
            calc_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            match = (calc_hash == expected_hash)
            results[name] = {
                "status": "PASS" if match else "MISMATCH",
                "path": str(path),
                "expected_sha256": expected_hash,
                "calculated_sha256": calc_hash,
                "match": match
            }

        all_ok = all(v.get("match", False) for v in results.values())
        self._integrity_verified = all_ok
        return {"all_checkpoints_verified": all_ok, "details": results}

    def _get_exp_d_model(self) -> ConvNeXtWaveletDetector:
        """Loads and caches the frozen Experiment D model."""
        if self._exp_d_model is None:
            if not self._integrity_verified:
                check = self.verify_checkpoints()
                if not check["all_checkpoints_verified"]:
                    raise RuntimeError(f"Checkpoint integrity verification failed: {check}")

            logger.info(f"Loading frozen Exp D ConvNeXt-Wavelet model from {EXPD_CHECKPOINT_PATH}...")
            model = ConvNeXtWaveletDetector(
                pretrained_backbone=False,
                freeze_backbone=True,
                dropout_p=0.2
            )
            checkpoint = torch.load(EXPD_CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self._exp_d_model = model
            logger.info("Exp D model loaded successfully.")
        return self._exp_d_model

    def run_face_vit_expert(self, video_path: Path) -> ExpertInferenceResult:
        """Executes Production Face ViT without modifying production code."""
        t0 = time.perf_counter()
        try:
            raw_res = predict_video_vit(str(video_path))
            t_ms = (time.perf_counter() - t0) * 1000.0

            label = raw_res.get("label", "ERROR")
            if label == "ERROR":
                return ExpertInferenceResult(
                    expert_id="production_face_vit",
                    expert_name="Production Vision Transformer (Face Specialist)",
                    status=ExpertStatus.ERROR,
                    raw_prediction=-1,
                    probability=None,
                    confidence=None,
                    face_detected=False,
                    evidence_status=raw_res.get("reliability_note", "ViT inference error"),
                    execution_time_ms=t_ms,
                    metadata=raw_res
                )
            elif label == "INSUFFICIENT_EVIDENCE":
                return ExpertInferenceResult(
                    expert_id="production_face_vit",
                    expert_name="Production Vision Transformer (Face Specialist)",
                    status=ExpertStatus.INSUFFICIENT_EVIDENCE,
                    raw_prediction=-1,
                    probability=None,
                    confidence=None,
                    face_detected=False,
                    evidence_status="No human faces detected in any sampled frame",
                    execution_time_ms=t_ms,
                    metadata=raw_res
                )
            else:
                p_fake = float(raw_res.get("fake_probability", 0.0))
                pred = 1 if label == "FAKE" else 0
                return ExpertInferenceResult(
                    expert_id="production_face_vit",
                    expert_name="Production Vision Transformer (Face Specialist)",
                    status=ExpertStatus.SUCCESS,
                    raw_prediction=pred,
                    probability=p_fake,
                    confidence=float(raw_res.get("confidence", 0.0)),
                    face_detected=True,
                    evidence_status=f"Valid facial sequence analyzed (faces={raw_res.get('face_count', 0)})",
                    execution_time_ms=t_ms,
                    metadata=raw_res
                )
        except Exception as e:
            t_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Face ViT execution failure: {e}", exc_info=True)
            return ExpertInferenceResult(
                expert_id="production_face_vit",
                expert_name="Production Vision Transformer (Face Specialist)",
                status=ExpertStatus.ERROR,
                raw_prediction=-1,
                probability=None,
                confidence=None,
                face_detected=False,
                evidence_status=f"Execution error: {str(e)}",
                execution_time_ms=t_ms,
                metadata={"error": str(e)}
            )

    def run_exp_d_expert(self, video_path: Path) -> ExpertInferenceResult:
        """Executes Phase 6.2B Experiment D ConvNeXt-Wavelet full-frame model."""
        t0 = time.perf_counter()
        try:
            model = self._get_exp_d_model()
            frames_tensor = extract_uniform_frames(video_path)
            frames_tensor = frames_tensor.to(self.device)

            with torch.no_grad():
                logits = model(frames_tensor)
                p_ai = F.softmax(logits, dim=1)[:, 1].mean().item()

            t_ms = (time.perf_counter() - t0) * 1000.0
            pred = 1 if p_ai >= 0.50 else 0

            return ExpertInferenceResult(
                expert_id="exp_d_wavelet",
                expert_name="ConvNeXt-Tiny + Haar Wavelet (Full-Frame Frequency Specialist)",
                status=ExpertStatus.SUCCESS,
                raw_prediction=pred,
                probability=round(p_ai, 4),
                confidence=round(abs(p_ai - 0.50) * 2.0, 4),
                face_detected=False,    # Full-frame model does not track faces
                evidence_status="Full-frame spatial & Haar wavelet frequency decomposition",
                execution_time_ms=t_ms,
                metadata={"num_frames": frames_tensor.shape[0]}
            )
        except Exception as e:
            t_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(f"Exp D execution failure: {e}", exc_info=True)
            return ExpertInferenceResult(
                expert_id="exp_d_wavelet",
                expert_name="ConvNeXt-Tiny + Haar Wavelet (Full-Frame Frequency Specialist)",
                status=ExpertStatus.ERROR,
                raw_prediction=-1,
                probability=None,
                confidence=None,
                face_detected=False,
                evidence_status=f"Execution error: {str(e)}",
                execution_time_ms=t_ms,
                metadata={"error": str(e)}
            )

    def analyze_video(
        self,
        video_path: str,
        policy: str = "POLICY_C",
        timeout_sec: float = DEFAULT_INFERENCE_TIMEOUT_SEC
    ) -> FusionDecisionResult:
        """
        Executes end-to-end two-expert staging evaluation under the specified policy.
        Results are strictly isolated and marked as EXPERIMENTAL_STAGING_RESULTS.
        """
        vpath = Path(video_path)
        t_start = time.perf_counter()

        with self.resource_mgr.execution_guard(timeout_sec=timeout_sec) as queue_wait_s:
            # Run Face ViT expert
            face_vit_res = self.run_face_vit_expert(vpath)

            # Run Exp D expert
            exp_d_res = self.run_exp_d_expert(vpath)

            total_ms = (time.perf_counter() - t_start) * 1000.0
            _, peak_ram = self.resource_mgr.get_memory_usage_mb()

            # Route through staging fusion router
            fusion_res = StagingFusionRouter.evaluate(
                face_vit_result=face_vit_res,
                exp_d_result=exp_d_res,
                policy=policy,
                total_latency_ms=round(total_ms, 2),
                peak_ram_mb=peak_ram,
                input_video_path=str(vpath)
            )

            return fusion_res
