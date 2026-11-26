"""
Standardized Video Preprocessing Pipeline for ST-SeqNet (Phase 7F.3)

Strict Requirements Enforced:
1. Standardized Decoding: Decodes videos into uncompressed RGB frames using OpenCV.
2. Uniform Temporal Sampling: Samples exactly 16 frames uniformly across the video timeline.
3. Fixed Spatial Resizing: Resizes each frame to exactly 224x224 with consistent bilinear interpolation.
4. Information Stripping: Strips container headers, codecs, frame rates, file paths, and category
   labels from the returned tensor representation to prevent low-level digital shortcut leakage.
5. Robust Fallback: Handles short videos (<16 frames) via cyclical temporal padding, corrupted
   frames via temporal imputation, and completely unreadable streams via zero-tensor fallback
   with explicit logging.
6. Configuration Versioning: Logs reproducible preprocessing parameters.
"""

import os
import cv2
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any, Union

PREPROCESSING_VERSION = "7F.3.0"

@dataclass(frozen=True)
class PreprocessingConfig:
    version: str = PREPROCESSING_VERSION
    num_frames: int = 16
    target_height: int = 224
    target_width: int = 224
    interpolation: str = "INTER_LINEAR"  # cv2.INTER_LINEAR
    to_rgb: bool = True
    normalize_imagenet: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_CONFIG = PreprocessingConfig()


class StandardVideoPreprocessor:
    """
    Standardized video preprocessor that decouples raw video container metadata
    from tensor representations for anti-shortcut AI video detection.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._cv2_interp = getattr(cv2, self.config.interpolation, cv2.INTER_LINEAR)
        self._mean = np.array(self.config.mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self.config.std, dtype=np.float32).reshape(1, 1, 3)

    def get_config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def sample_frame_indices(self, total_frames: int) -> np.ndarray:
        """
        Calculates 16 uniformly spaced integer frame indices across video duration.
        """
        if total_frames <= 0:
            return np.zeros(self.config.num_frames, dtype=int)
        if total_frames < self.config.num_frames:
            # Short video: repeat frames cyclically
            indices = np.arange(total_frames)
            repeats = int(np.ceil(self.config.num_frames / total_frames))
            tiled = np.tile(indices, repeats)
            return tiled[:self.config.num_frames]
        # Uniform sampling across [0, total_frames - 1]
        return np.linspace(0, total_frames - 1, self.config.num_frames, dtype=int)

    def decode_video(
        self, video_path: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Decodes video file into an array of shape (16, 224, 224, 3) uint8 RGB frames.
        Returns (frames_array, status_dict).
        Does NOT return or expose path, codec, or container metadata to the model.
        """
        status: Dict[str, Any] = {
            "success": False,
            "total_frames_reported": 0,
            "frames_decoded": 0,
            "padding_applied": False,
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
        status["total_frames_reported"] = total_frames

        # If container frame count is negative or zero, count quickly via grab
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

        target_indices = self.sample_frame_indices(total_frames)
        selected_raw = []
        last_valid_frame = None

        # Direct seeking loop
        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                selected_raw.append(frame)
                last_valid_frame = frame
            elif last_valid_frame is not None:
                selected_raw.append(last_valid_frame)
                status["padding_applied"] = True
            else:
                selected_raw.append(None)

        cap.release()

        # Handle any initial None frames
        if any(f is None for f in selected_raw):
            first_valid = next((f for f in selected_raw if f is not None), None)
            if first_valid is None:
                status["error_msg"] = "All decoded target frames were null"
                return None, status
            selected_raw = [f if f is not None else first_valid for f in selected_raw]
            status["padding_applied"] = True

        status["frames_decoded"] = len(selected_raw)

        # Process each frame: resize to 224x224 and BGR -> RGB
        processed_frames = []
        for frame in selected_raw:
            resized = cv2.resize(
                frame,
                (self.config.target_width, self.config.target_height),
                interpolation=self._cv2_interp,
            )
            if self.config.to_rgb:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                rgb = resized
            processed_frames.append(rgb)

        frames_array = np.stack(processed_frames, axis=0)  # Shape: (16, 224, 224, 3) uint8
        status["success"] = True
        return frames_array, status

    def frames_to_tensor(
        self, frames_array: np.ndarray, apply_normalization: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Converts (16, 224, 224, 3) uint8 array to float PyTorch tensor:
        Shape: (16, 3, 224, 224)
        """
        norm = self.config.normalize_imagenet if apply_normalization is None else apply_normalization
        float_frames = frames_array.astype(np.float32) / 255.0

        if norm:
            float_frames = (float_frames - self._mean) / self._std

        tensor = torch.from_numpy(float_frames).permute(0, 3, 1, 2)
        return tensor

    def process_to_tensor(
        self, video_path: str, apply_normalization: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        End-to-end processing from video file directly to sanitized (16, 3, 224, 224) tensor.
        """
        frames_array, status = self.decode_video(video_path)
        if not status["success"] or frames_array is None:
            zero_tensor = torch.zeros(
                (self.config.num_frames, 3, self.config.target_height, self.config.target_width),
                dtype=torch.float32,
            )
            return zero_tensor, status

        tensor = self.frames_to_tensor(frames_array, apply_normalization=apply_normalization)
        return tensor, status
