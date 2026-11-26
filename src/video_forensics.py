"""
Multi-Signal Video Forensics & Whole-Frame Analysis Engine
==========================================================
Module: src/video_forensics.py
Implements deterministic, interpretable whole-frame video forensics:
  1. 2D FFT/DCT radial power spectral analysis and high-frequency energy ratio.
  2. Multi-scale Laplacian texture energy and surface smoothing analysis.
  3. Inter-frame temporal motion flux and morphing consistency.
  4. Compression and video quality metrics (blockiness boundary discontinuity).
  5. Biometric face presence and coverage tracking.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection_engine import (
    DetectionVerdict,
    EvidenceQualityTier,
    ForensicSignal,
    UnifiedDetectionResult,
    build_video_detection_result,
    format_percentage,
    serialize_forensic_signal,
    serialize_unified_result,
    to_json_safe,
)


@dataclass
class WholeFrameMetrics:
    """Quantitative measurements extracted from video frames."""
    # Spectral
    spectral_slope: float = 0.0
    hf_energy_ratio: float = 0.0
    spectral_anomaly_score: float = 0.0
    
    # Texture
    laplacian_variance: float = 0.0
    normalized_laplacian: float = 0.0
    texture_smoothing_score: float = 0.0

    # Temporal
    mean_flux: float = 0.0
    flux_std: float = 0.0
    temporal_anomaly_score: float = 0.0

    # Quality & Compression
    width: int = 0
    height: int = 0
    aspect_ratio: float = 1.0
    blockiness_index: float = 1.0
    is_compressed: bool = False

    # Aggregate Forensic Score
    synthetic_generative_score: float = 0.5


def compute_spectral_analysis(frames_gray: List[np.ndarray]) -> Tuple[float, float, float, str, str]:
    """
    Compute radial power spectrum slope and high-frequency band energy using 2D FFT.
    Natural optical photography obeys power-law falloff: P(f) ~ 1 / f^alpha with alpha in [1.6, 2.4].
    Synthetic generative diffusion models often exhibit:
      - Over-smoothed VAE output (steep slope < -2.8 or very low high-frequency ratio < 0.005).
      - Transposed convolution checkerboards (slope > -1.2 or high-frequency spikes > 0.08).
    """
    if not frames_gray:
        return 0.0, 0.0, 0.5, "NEUTRAL", "WEAK"

    slopes: List[float] = []
    hf_ratios: List[float] = []
    h, w = frames_gray[0].shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max(min(cx, cy), 1)

    bins = np.linspace(0.05, 0.45, 15)

    for f in frames_gray:
        f_shift = np.fft.fftshift(np.fft.fft2(f))
        mag = np.abs(f_shift) ** 2
        total_e = float(np.sum(mag)) + 1e-9

        # High-frequency radial band (r > 0.35)
        hf_e = float(np.sum(mag[r > 0.35]))
        hf_ratios.append(hf_e / total_e)

        # Radial bin averages for spectral slope
        b_means: List[float] = []
        b_cents: List[float] = []
        for i in range(len(bins) - 1):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if np.any(mask):
                b_means.append(float(np.mean(mag[mask])))
                b_cents.append(float(0.5 * (bins[i] + bins[i + 1])))

        if len(b_cents) >= 4:
            slope, _ = np.polyfit(np.log(b_cents), np.log(np.array(b_means) + 1e-12), 1)
            slopes.append(float(slope))

    avg_slope = float(np.mean(slopes)) if slopes else -2.0
    avg_hf = float(np.mean(hf_ratios)) if hf_ratios else 0.02

    # Calibrated distance from natural camera range (natural camera typical: hf in [0.012, 0.045])
    if avg_hf < 0.0035:
        # Severe high-frequency deficit (characteristic of latent diffusion / low-pass smoothing)
        anomaly_score = min(0.95, 0.65 + (0.0035 - avg_hf) * 80.0)
        indication = "SUPPORTS_AI"
        strength = "STRONG"
    elif avg_hf < 0.0080:
        anomaly_score = 0.60
        indication = "SUPPORTS_AI"
        strength = "MODERATE"
    elif avg_hf > 0.12:
        # Extreme grid noise
        anomaly_score = 0.70
        indication = "SUPPORTS_AI"
        strength = "MODERATE"
    else:
        # Natural optical distribution
        anomaly_score = max(0.10, 0.40 - avg_hf * 5.0)
        indication = "SUPPORTS_HUMAN"
        strength = "MODERATE"

    return round(avg_slope, 2), round(avg_hf, 4), round(anomaly_score, 3), indication, strength


def compute_spatial_texture_variance(frames_gray: List[np.ndarray]) -> Tuple[float, float, float, str, str]:
    """
    Compute Laplacian variance and luminance-normalized texture density.
    Natural video features authentic sensor noise and micro-textures.
    Synthetic diffusion frequently displays unnaturally flattened surfaces.
    """
    if not frames_gray:
        return 0.0, 0.0, 0.5, "NEUTRAL", "WEAK"

    lap_vars: List[float] = []
    norm_laps: List[float] = []

    for f in frames_gray:
        lv = float(cv2.Laplacian(f, cv2.CV_64F).var())
        mean_lum = float(np.mean(f)) + 1e-4
        lap_vars.append(lv)
        norm_laps.append(lv / mean_lum)

    avg_var = float(np.mean(lap_vars))
    avg_norm = float(np.mean(norm_laps))

    # Calibrate texture score
    if avg_var < 50.0:
        smoothing_score = 0.75
        indication = "SUPPORTS_AI"
        strength = "MODERATE"
    elif avg_var > 150.0:
        smoothing_score = 0.20
        indication = "SUPPORTS_HUMAN"
        strength = "MODERATE"
    else:
        smoothing_score = 0.45
        indication = "NEUTRAL"
        strength = "WEAK"

    return round(avg_var, 1), round(avg_norm, 2), round(smoothing_score, 3), indication, strength


def compute_temporal_frame_flux(frames_gray: List[np.ndarray]) -> Tuple[float, float, float, str, str]:
    """
    Compute inter-frame motion flux and flux variance.
    Natural stationary/panning video exhibits smooth, bounded temporal changes.
    Diffusion generative video often suffers from latent morphing flux or unnatural frame jitter.
    """
    if len(frames_gray) < 2:
        return 0.0, 0.0, 0.5, "NEUTRAL", "WEAK"

    fluxes: List[float] = []
    for i in range(len(frames_gray) - 1):
        diff = cv2.absdiff(frames_gray[i + 1], frames_gray[i])
        fluxes.append(float(np.mean(diff)))

    avg_flux = float(np.mean(fluxes))
    flux_std = float(np.std(fluxes)) if len(fluxes) > 1 else 0.0

    # Extremely high flux (> 25.0) indicates synthesis flicker or fast morphing
    # Very low flux (< 0.5) indicates static/talking head video
    if avg_flux > 12.0:
        temporal_score = 0.70
        indication = "SUPPORTS_AI"
        strength = "MODERATE"
    elif avg_flux > 3.0 and flux_std > 2.5:
        temporal_score = 0.60
        indication = "SUPPORTS_AI"
        strength = "WEAK"
    elif avg_flux < 1.0:
        temporal_score = 0.30
        indication = "SUPPORTS_HUMAN"
        strength = "WEAK"
    else:
        temporal_score = 0.45
        indication = "NEUTRAL"
        strength = "WEAK"

    return round(avg_flux, 2), round(flux_std, 2), round(temporal_score, 3), indication, strength


def compute_compression_and_quality(
    frames_gray: List[np.ndarray],
    file_size_bytes: int = 0
) -> Tuple[float, bool, str, str]:
    """
    Evaluate H.264/H.265 compression blockiness along 8x8 macroblock boundaries.
    Prevents false alarms on heavily compressed web videos where DCT quantization
    might otherwise be confused with diffusion smoothing.
    """
    if not frames_gray:
        return 1.0, False, "NEUTRAL", "WEAK"

    block_ratios: List[float] = []
    for f in frames_gray:
        c1 = f[:, 7::8].astype(float)
        c2 = f[:, 8::8].astype(float)
        m = min(c1.shape[1], c2.shape[1])
        if m > 0:
            diff_bound = float(np.mean(np.abs(c1[:, :m] - c2[:, :m])))
            c_in1 = f[:, 6::8].astype(float)
            c_in2 = f[:, 7::8].astype(float)
            m_in = min(c_in1.shape[1], c_in2.shape[1])
            diff_inner = float(np.mean(np.abs(c_in1[:, :m_in] - c_in2[:, :m_in])))
            block_ratios.append(diff_bound / max(diff_inner, 1e-3))

    avg_block = float(np.mean(block_ratios)) if block_ratios else 1.0
    is_compressed = avg_block > 1.10

    if is_compressed:
        indication = "NEUTRAL"
        strength = "MODERATE"
    else:
        indication = "SUPPORTS_HUMAN"
        strength = "WEAK"

    return round(avg_block, 3), is_compressed, indication, strength


def extract_comprehensive_whole_frame_forensics(
    video_path: str,
    max_frames: int = 8,
    face_count: int = 0
) -> Tuple[WholeFrameMetrics, List[ForensicSignal]]:
    """
    Extract multi-modal forensic measurements and format them into structured ForensicSignals.
    Normalizes spatial resolution to (480, 270) to ensure consistent frequency scales across
    different source resolutions, aspect ratios, and compression levels.
    """
    if not os.path.exists(video_path):
        return WholeFrameMetrics(), []

    file_size = os.path.getsize(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return WholeFrameMetrics(), []

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 480)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 270)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 16)
    aspect_ratio = round(orig_w / max(orig_h, 1), 2)

    # Uniform sampling
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = list(range(max(total_frames, 1)))

    frames_gray: List[np.ndarray] = []
    current_idx = 0
    sampled_set = set(indices)

    while cap.isOpened() and len(frames_gray) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in sampled_set:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Standardize resolution for scale-invariant spectral & Laplacian evaluation
            gray_std = cv2.resize(gray, (480, 270), interpolation=cv2.INTER_AREA)
            frames_gray.append(gray_std)
        current_idx += 1

    cap.release()

    if not frames_gray:
        return WholeFrameMetrics(width=orig_w, height=orig_h, aspect_ratio=aspect_ratio), []

    # 1. Spectral Analysis
    slope, hf_ratio, spec_score, spec_ind, spec_str = compute_spectral_analysis(frames_gray)

    # 2. Spatial Texture Analysis
    lap_var, norm_lap, text_score, text_ind, text_str = compute_spatial_texture_variance(frames_gray)

    # 3. Temporal Consistency
    flux, flux_std, temp_score, temp_ind, temp_str = compute_temporal_frame_flux(frames_gray)

    # 4. Compression & Blockiness
    block_idx, is_comp, comp_ind, comp_str = compute_compression_and_quality(frames_gray, file_size)

    # 5. Aggregate Generative Anomaly Score (Calibrated Multi-Evidence Fusion)
    # If heavily compressed, down-weight high-frequency and texture smoothing
    if is_comp:
        w_spec = 0.35
        w_text = 0.20
        w_temp = 0.45
    else:
        w_spec = 0.50
        w_text = 0.30
        w_temp = 0.20

    raw_gen_score = w_spec * spec_score + w_text * text_score + w_temp * temp_score
    gen_score = float(np.clip(raw_gen_score, 0.05, 0.95))

    metrics = WholeFrameMetrics(
        spectral_slope=slope,
        hf_energy_ratio=hf_ratio,
        spectral_anomaly_score=spec_score,
        laplacian_variance=lap_var,
        normalized_laplacian=norm_lap,
        texture_smoothing_score=text_score,
        mean_flux=flux,
        flux_std=flux_std,
        temporal_anomaly_score=temp_score,
        width=orig_w,
        height=orig_h,
        aspect_ratio=aspect_ratio,
        blockiness_index=block_idx,
        is_compressed=is_comp,
        synthetic_generative_score=round(gen_score, 3)
    )

    signals: List[ForensicSignal] = []

    # Biometric Coverage
    face_pct = round((face_count / max(max_frames, 1)) * 100, 1)
    if face_count >= 6:
        signals.append(ForensicSignal(
            name="Biometric Face Coverage",
            category="biometric",
            description=f"High biometric evidence: {face_count}/{max_frames} frames ({face_pct}%) contain trackable faces.",
            value=f"{face_count}/{max_frames} frames",
            indication="NEUTRAL",
            strength="STRONG",
        ))
    elif face_count >= 3:
        signals.append(ForensicSignal(
            name="Biometric Face Coverage",
            category="biometric",
            description=f"Partial biometric evidence: {face_count}/{max_frames} frames ({face_pct}%) contain trackable faces.",
            value=f"{face_count}/{max_frames} frames",
            indication="NEUTRAL",
            strength="MODERATE",
        ))
    else:
        signals.append(ForensicSignal(
            name="Biometric Face Coverage",
            category="biometric",
            description=f"Faceless scene ({face_count}/{max_frames} frames). Facial patch attention is disabled; whole-frame forensics active.",
            value=f"{face_count}/{max_frames} frames",
            indication="NEUTRAL",
            strength="WEAK",
        ))

    # Spatial Texture
    signals.append(ForensicSignal(
        name="Spatial Texture Sharpness",
        category="spatial_texture",
        description="Measures surface grain via Laplacian variance. AI generators often display unnatural micro-smoothing.",
        value=lap_var,
        indication=text_ind,
        strength=text_str,
    ))

    # Frequency Domain
    signals.append(ForensicSignal(
        name="Spectral High-Frequency Energy",
        category="frequency_domain",
        description=f"Ratio of high-frequency 2D Fourier energy (HF ratio: {hf_ratio}, slope: {slope}). Low ratios indicate generative diffusion smoothing.",
        value=f"{round(hf_ratio * 100, 2)}% HF energy",
        indication=spec_ind,
        strength=spec_str,
    ))

    # Temporal Consistency
    signals.append(ForensicSignal(
        name="Temporal Frame Consistency",
        category="temporal",
        description=f"Inter-frame flux variance (flux: {flux}, std: {flux_std}). High morphing flux is common in latent diffusion videos.",
        value=flux,
        indication=temp_ind,
        strength=temp_str,
    ))

    # Compression Quality
    comp_desc = "Standard/Clean Video" if not is_comp else f"High Codec Compression (Blockiness: {block_idx})"
    signals.append(ForensicSignal(
        name="Compression Quality Metric",
        category="compression",
        description="Evaluates 8x8 macroblock boundary discontinuity to prevent codec false alarms on web video.",
        value=comp_desc,
        indication=comp_ind,
        strength=comp_str,
    ))

    return metrics, signals


def extract_video_forensic_signals(
    video_path: str,
    max_frames: int = 8,
    face_count: int = 0
) -> List[ForensicSignal]:
    """Compatibility wrapper returning List[ForensicSignal]."""
    _, signals = extract_comprehensive_whole_frame_forensics(video_path, max_frames=max_frames, face_count=face_count)
    return signals
