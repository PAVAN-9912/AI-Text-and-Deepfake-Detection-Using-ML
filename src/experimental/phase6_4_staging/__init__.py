"""
Phase 6.4 Staging Package
=========================
Isolated two-expert video fusion staging implementation.
"""

from .staging_result_schema import (
    ExpertInferenceResult,
    FusionDecisionResult,
    FusionVerdict,
    ExpertStatus
)
from .staging_policy_config import (
    POLICY_C_CONFIG,
    POLICY_D_CONFIG,
    VIT_CHECKPOINT_PATH,
    EXPD_CHECKPOINT_PATH
)
from .staging_resource_manager import ResourceManager
from .staging_video_preprocessor import extract_uniform_frames, VideoPreprocessingError
from .staging_fusion_router import StagingFusionRouter
from .staging_video_engine import StagingVideoEngine

__all__ = [
    "ExpertInferenceResult",
    "FusionDecisionResult",
    "FusionVerdict",
    "ExpertStatus",
    "POLICY_C_CONFIG",
    "POLICY_D_CONFIG",
    "VIT_CHECKPOINT_PATH",
    "EXPD_CHECKPOINT_PATH",
    "ResourceManager",
    "extract_uniform_frames",
    "VideoPreprocessingError",
    "StagingFusionRouter",
    "StagingVideoEngine",
]
