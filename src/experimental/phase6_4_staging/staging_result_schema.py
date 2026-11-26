"""
Staging Result Schema
=====================
Defines structured data contracts for expert model outputs and fusion decisions
under the Phase 6.4 staging architecture.

All outputs are strictly tagged with STAGING_MODE = 'EXPERIMENTAL_STAGING_RESULTS'.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class FusionVerdict(str, Enum):
    LIKELY_HUMAN = "LIKELY_HUMAN"
    AI_GENERATED = "AI_GENERATED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ExpertStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ERROR = "ERROR"


@dataclass
class ExpertInferenceResult:
    expert_id: str                      # "production_face_vit" or "exp_d_wavelet"
    expert_name: str                    # Human-readable name
    status: ExpertStatus
    raw_prediction: int                 # 0 (Genuine), 1 (AI/Fake), -1 (Abstain/Error)
    probability: Optional[float]        # Fake/AI probability in [0.0, 1.0]
    confidence: Optional[float]         # Operational confidence margin
    face_detected: bool                 # True if subject faces were extracted
    evidence_status: str                # Qualitative description of evidence
    execution_time_ms: float            # Inference latency in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class FusionDecisionResult:
    staging_mode: str                   # Constant: "EXPERIMENTAL_STAGING_RESULTS"
    policy_name: str                    # "POLICY_C" or "POLICY_D"
    verdict: FusionVerdict              # LIKELY_HUMAN, AI_GENERATED, INCONCLUSIVE
    raw_prediction: int                 # 0 (Human), 1 (AI), -1 (Inconclusive)
    calibrated_score: Optional[float]   # Fused score (not to be conflated with real-world AI likelihood)
    decision_rule: str                  # Formal rule ID (e.g. "C1_FACELESS_DELEGATION")
    explanation: str                    # Clear audit justification
    is_abstained: bool                  # True if output is INCONCLUSIVE
    domain_conflict: bool               # True if experts presented contradictory decisive signals
    face_vit_result: ExpertInferenceResult
    exp_d_result: ExpertInferenceResult
    total_latency_ms: float             # End-to-end staging execution time in ms
    peak_ram_mb: float                  # Peak RAM observed during execution
    input_video_path: str               # Filepath of the analyzed video

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["face_vit_result"] = self.face_vit_result.to_dict()
        d["exp_d_result"] = self.exp_d_result.to_dict()
        return d
