"""
Staging Video Preprocessor
==========================
Handles secure video frame extraction, normalization, and tensor formatting
for Phase 6.2B Experiment D ConvNeXt-Wavelet model.

Guarantees:
- Safe file-descriptor release.
- Corrupted file detection with structured errors.
- Strict dimension and ImageNet normalization compliance.
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import torch
import logging

from .staging_policy_config import (
    UNIFORM_SAMPLED_FRAMES,
    FRAME_SIZE,
    MAX_INPUT_FILE_SIZE_MB
)

logger = logging.getLogger("staging_video_preprocessor")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class VideoPreprocessingError(Exception):
    """Raised when video extraction or validation fails."""
    pass


def extract_uniform_frames(
    filepath: Path,
    num_frames: int = UNIFORM_SAMPLED_FRAMES,
    target_size: Tuple[int, int] = FRAME_SIZE
) -> torch.Tensor:
    """
    Extracts uniformly spaced normalized frame tensors from a video file.
    Output shape: (num_frames, 3, target_size[0], target_size[1])
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Video file not found: {filepath}")

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_INPUT_FILE_SIZE_MB:
        raise VideoPreprocessingError(
            f"File size {file_size_mb:.2f}MB exceeds safe threshold of {MAX_INPUT_FILE_SIZE_MB}MB"
        )

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        cap.release()
        raise VideoPreprocessingError(f"OpenCV cannot open video stream: {filepath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise VideoPreprocessingError(f"Invalid frame count ({total_frames}) for video: {filepath}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames_list = []

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback to mid-point frame if seek fails
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise VideoPreprocessingError(f"Corrupted frame encountered at index {idx} in {filepath}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            frame_norm = (frame_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1))
            frames_list.append(frame_tensor)
    finally:
        cap.release()

    if len(frames_list) != num_frames:
        raise VideoPreprocessingError(
            f"Expected {num_frames} frames, but only extracted {len(frames_list)}"
        )

    return torch.stack(frames_list, dim=0)
