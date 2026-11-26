"""
Unified AI Content Detection Engine & Result Formatting Module
==============================================================
Defines the standard percentage-based result contract, evidence quality tiers,
and forensic signal schema for both text and video detection pipelines.

Invariants:
  1. Distinguishes estimated model scores from mathematically calibrated probabilities.
  2. Communicates clear 3-way verdicts: AI_GENERATED, LIKELY_HUMAN, or INCONCLUSIVE.
  3. Provides structured evidence quality tiers: HIGH, MODERATE, LIMITED, or INSUFFICIENT.
  4. Exposes interpretable forensic signals and explicit platform limitations.
  5. Implements explicit, robust JSON serialization safe for NumPy types, Enums, and dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DetectionVerdict(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    LIKELY_HUMAN = "LIKELY_HUMAN"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceQualityTier(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LIMITED = "LIMITED"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class ForensicSignal:
    """A discrete interpretable detection signal."""
    name: str
    category: str  # e.g., "linguistic", "biometric", "spatial_texture", "frequency_domain", "temporal"
    description: str
    value: Any
    indication: str  # "SUPPORTS_AI", "SUPPORTS_HUMAN", "NEUTRAL"
    strength: str    # "STRONG", "MODERATE", "WEAK"


@dataclass
class UnifiedDetectionResult:
    """The standard platform response format for all media modalities."""
    media_type: str  # "text" or "video"
    verdict: DetectionVerdict
    ai_likelihood_percentage: Optional[float]
    confidence_percentage: Optional[float]
    evidence_quality: EvidenceQualityTier
    summary: str
    forensic_signals: List[ForensicSignal] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to clean JSON-serializable dictionary."""
        return {
            "media_type": self.media_type,
            "verdict": self.verdict.value,
            "ai_likelihood_percentage": self.ai_likelihood_percentage,
            "confidence_percentage": self.confidence_percentage,
            "evidence_quality": self.evidence_quality.value,
            "summary": self.summary,
            "forensic_signals": [serialize_forensic_signal(s) for s in self.forensic_signals],
            "limitations": self.limitations,
            "metadata": to_json_safe(self.metadata),
        }


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert arbitrary objects, numpy types, enums, dataclasses,
    and missing values into pure JSON-serializable standard Python types.
    """
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(round(obj, 4))
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return to_json_safe(asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_json_safe(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return {str(k): to_json_safe(v) for k, v in obj.__dict__.items() if not str(k).startswith("_")}
    return str(obj)


def serialize_forensic_signal(signal: Any) -> Dict[str, Any]:
    """Explicitly serialize a ForensicSignal or dict into a guaranteed JSON-compliant dict."""
    if signal is None:
        return {}
    if isinstance(signal, dict):
        d = signal
    elif hasattr(signal, "__dataclass_fields__"):
        d = asdict(signal)
    elif hasattr(signal, "__dict__"):
        d = signal.__dict__
    else:
        d = {
            "name": getattr(signal, "name", "unknown_signal"),
            "category": getattr(signal, "category", "general"),
            "description": getattr(signal, "description", ""),
            "value": getattr(signal, "value", None),
            "indication": getattr(signal, "indication", "NEUTRAL"),
            "strength": getattr(signal, "strength", "WEAK"),
        }

    return {
        "name": str(d.get("name", "")),
        "category": str(d.get("category", "")),
        "description": str(d.get("description", "")),
        "value": to_json_safe(d.get("value")),
        "indication": str(to_json_safe(d.get("indication", "NEUTRAL"))),
        "strength": str(to_json_safe(d.get("strength", "WEAK"))),
    }


def serialize_unified_result(result: Any) -> Dict[str, Any]:
    """Guaranteed JSON-serializable dictionary for UnifiedDetectionResult."""
    if result is None:
        return {}
    if hasattr(result, "to_dict") and callable(result.to_dict):
        raw = result.to_dict()
    elif isinstance(result, dict):
        raw = result
    elif hasattr(result, "__dict__"):
        raw = result.__dict__
    else:
        return {}
    return to_json_safe(raw)


def format_percentage(value: Optional[float], decimals: int = 1) -> Optional[float]:
    """Format and bound float values between 0.0% and 100.0%."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    val = float(value)
    if val <= 1.0:
        val = val * 100.0
    val = max(0.0, min(100.0, val))
    return round(val, decimals)


def build_text_detection_result(
    raw_ai_prob: Optional[float],
    word_count: int,
    linguistic_signals: List[ForensicSignal],
    length_group: str,
    threshold: float,
    is_code: bool = False,
    is_multilingual: bool = False,
) -> UnifiedDetectionResult:
    """
    Standardize text detector outputs into the unified percentage-based format.
    Distinguishes high-confidence long text from short/ambiguous text.
    """
    limitations: List[str] = [
        "Text detection evaluates statistical stylistic patterns and vocabulary distributions. Highly edited, paraphrased, or technical text may produce ambiguous scores."
    ]

    # Case 1: Ultra-short text (<= 2 words) -> Inconclusive
    if word_count <= 2 or raw_ai_prob is None:
        return UnifiedDetectionResult(
            media_type="text",
            verdict=DetectionVerdict.INCONCLUSIVE,
            ai_likelihood_percentage=None,
            confidence_percentage=None,
            evidence_quality=EvidenceQualityTier.INSUFFICIENT,
            summary="Input text contains fewer than 3 words. There is insufficient linguistic context for reliable statistical evaluation.",
            forensic_signals=linguistic_signals,
            limitations=limitations + ["Minimum 3 words required for initial lexical analysis; >= 30 words recommended for high reliability."],
            metadata={
                "word_count": word_count,
                "length_group": length_group,
                "threshold": round(float(threshold), 2)
            }
        )

    ai_pct = format_percentage(raw_ai_prob)

    # Determine Evidence Quality Tier
    if word_count <= 25:
        evidence_quality = EvidenceQualityTier.LIMITED
        limitations.append("Short text (< 25 words) provides limited stylistic sample size. Confidence is discounted accordingly.")
        quality_factor = 0.70
    elif word_count <= 60 or is_multilingual or is_code:
        evidence_quality = EvidenceQualityTier.MODERATE
        if is_multilingual:
            limitations.append("Non-English or multilingual elements detected. Stylistic distributions may exhibit domain shifts.")
        if is_code:
            limitations.append("Code syntax or structured programming constructs detected. Stylistic entropy patterns differ from prose.")
        quality_factor = 0.85
    else:
        evidence_quality = EvidenceQualityTier.HIGH
        quality_factor = 0.98

    # Map Verdict and calculate confidence
    thresh_pct = format_percentage(threshold)
    dist_from_threshold = abs(ai_pct - thresh_pct)

    # Equivocal boundary handling (within 5% of decision threshold)
    if dist_from_threshold < 5.0:
        verdict = DetectionVerdict.INCONCLUSIVE
        base_conf = 45.0 + (dist_from_threshold / 5.0) * 10.0
        conf_pct = round(base_conf * quality_factor, 1)
        summary = f"Text exhibits mixed stylistic signals with estimated AI-likelihood ({ai_pct}%) near the decision boundary ({thresh_pct}%). The result is borderline and inconclusive."
    elif ai_pct >= thresh_pct:
        verdict = DetectionVerdict.AI_GENERATED
        # Confidence reflects how decisively the model exceeds threshold, scaled by evidence quality
        margin_ratio = min(1.0, (ai_pct - thresh_pct) / max(100.0 - thresh_pct, 1.0))
        base_conf = 55.0 + margin_ratio * 40.0
        conf_pct = round(base_conf * quality_factor, 1)
        summary = f"Text displays pronounced statistical markers characteristic of AI generation (estimated AI-likelihood: {ai_pct}%)."
    else:
        verdict = DetectionVerdict.LIKELY_HUMAN
        # Confidence reflects how decisively the model is below threshold, scaled by evidence quality
        margin_ratio = min(1.0, (thresh_pct - ai_pct) / max(thresh_pct, 1.0))
        base_conf = 55.0 + margin_ratio * 40.0
        conf_pct = round(base_conf * quality_factor, 1)
        summary = f"Text displays natural stylistic entropy, personal voice, and variability consistent with human writing (estimated AI-likelihood: {ai_pct}%)."

    return UnifiedDetectionResult(
        media_type="text",
        verdict=verdict,
        ai_likelihood_percentage=ai_pct,
        confidence_percentage=conf_pct,
        evidence_quality=evidence_quality,
        summary=summary,
        forensic_signals=linguistic_signals,
        limitations=limitations,
        metadata={
            "word_count": word_count,
            "length_group": length_group,
            "decision_threshold_pct": thresh_pct,
            "is_code": bool(is_code),
            "is_multilingual": bool(is_multilingual),
        }
    )


def build_video_detection_result(
    raw_fake_prob: Optional[float],
    frames_analyzed: int,
    valid_face_frames: int,
    resolution: str,
    biometric_signals: List[ForensicSignal],
    primary_expert: str,
    router_reason: str,
    threshold: float = 0.475,
    swin_suppressed: bool = True,
) -> UnifiedDetectionResult:
    """
    Standardize video detector outputs into the unified percentage-based format.
    Explicitly communicates biometric evidence depth and faceless limitations.
    Adheres strictly to the production Face ViT decision boundary.
    """
    limitations: List[str] = [
        "Video forensics evaluates facial boundary manipulation, spatial frequency textures, and temporal consistency. Compression artifacts from social media transcode may impact signal fidelity."
    ]

    # Case 1: Zero or Inadequate Faces (< 1 face frame)
    if valid_face_frames == 0 or raw_fake_prob is None:
        limitations.append("Facial Vision Transformer analysis requires detectable human faces. Faceless videos cannot be certified by facial patch attention; whole-frame spatial/frequency signals are advisory.")
        return UnifiedDetectionResult(
            media_type="video",
            verdict=DetectionVerdict.INCONCLUSIVE,
            ai_likelihood_percentage=None,
            confidence_percentage=None,
            evidence_quality=EvidenceQualityTier.INSUFFICIENT,
            summary=f"No detectable human faces found ({valid_face_frames}/{frames_analyzed} frames). Facial Vision Transformer analysis requires detectable facial regions.",
            forensic_signals=biometric_signals,
            limitations=limitations,
            metadata={
                "frames_analyzed": frames_analyzed,
                "valid_face_frames": valid_face_frames,
                "resolution": resolution,
                "primary_expert": primary_expert,
                "swin_suppressed": swin_suppressed,
            }
        )

    fake_pct = format_percentage(raw_fake_prob)

    # Determine Evidence Quality Tier based on valid face tracks
    if valid_face_frames >= 6:
        evidence_quality = EvidenceQualityTier.HIGH
        quality_factor = 0.98
    elif valid_face_frames >= 3:
        evidence_quality = EvidenceQualityTier.MODERATE
        quality_factor = 0.85
        limitations.append(f"Moderate facial evidence ({valid_face_frames}/{frames_analyzed} frames). Partial head turn or occlusion may slightly limit multi-frame temporal attention.")
    else:
        evidence_quality = EvidenceQualityTier.LIMITED
        quality_factor = 0.70
        limitations.append(f"Limited facial evidence ({valid_face_frames}/{frames_analyzed} frames). Only sparse facial crops were available for analysis.")

    # Face ViT Decision Rule:
    # Model produces P(Real) and P(Fake) = 1.0 - P(Real).
    # Calibrated decision threshold: REAL if P(Real) >= threshold.
    # Therefore, FAKE if P(Fake) > (1.0 - threshold).
    thresh_real = float(threshold)
    thresh_fake_pct = format_percentage(1.0 - thresh_real)
    raw_real_prob = 1.0 - (float(raw_fake_prob) if raw_fake_prob <= 1.0 else float(raw_fake_prob) / 100.0)

    is_real = raw_real_prob >= thresh_real

    if not is_real:
        verdict = DetectionVerdict.AI_GENERATED
        # Confidence reflects distance above fake threshold, scaled by face coverage
        margin = abs(fake_pct - thresh_fake_pct) / max(100.0 - thresh_fake_pct, 1.0)
        base_conf = 55.0 + min(40.0, margin * 45.0)
        conf_pct = round(base_conf * quality_factor, 1)
        summary = f"Biometric patch attention detected structural manipulation seams and facial boundary artifacts consistent with deepfake synthesis (estimated AI-likelihood: {fake_pct}%)."
    else:
        verdict = DetectionVerdict.LIKELY_HUMAN
        # Confidence reflects distance below fake threshold, scaled by face coverage
        margin = abs(thresh_fake_pct - fake_pct) / max(thresh_fake_pct, 1.0)
        base_conf = 55.0 + min(40.0, margin * 45.0)
        conf_pct = round(base_conf * quality_factor, 1)
        summary = f"Biometric features display natural micro-vascular texture, coherent landmark geometry, and authentic spatial continuity (estimated AI-likelihood: {fake_pct}%)."

    return UnifiedDetectionResult(
        media_type="video",
        verdict=verdict,
        ai_likelihood_percentage=fake_pct,
        confidence_percentage=conf_pct,
        evidence_quality=evidence_quality,
        summary=summary,
        forensic_signals=biometric_signals,
        limitations=limitations,
        metadata={
            "frames_analyzed": frames_analyzed,
            "valid_face_frames": valid_face_frames,
            "resolution": resolution,
            "primary_expert": primary_expert,
            "swin_suppressed": swin_suppressed,
            "threshold_fake_pct": thresh_fake_pct,
        }
    )
