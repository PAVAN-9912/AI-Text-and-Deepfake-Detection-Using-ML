"""
Anti-Shortcut Data Augmentation Pipeline for Video Forensic Training (Phase 7F.3)

Anti-Shortcut Rationale:
Phase 7F.2 demonstrated that 100% of WebM/VP9 videos are Genuine, while 82% of AI videos are H.264
(with 0 Genuine H.264). Without anti-shortcut augmentation, networks learn DCT quantization and
compression artifacts rather than generative spatio-temporal artifacts.

This module applies controlled video-level transformations:
1. Random JPEG Recompression (Quality 50-95): Masking native container compression artifacts.
2. Gaussian Blur (kernel 3x3 or 5x5, sigma 0.2-1.5): Attenuating high-frequency container noise.
3. Color & Brightness Variation: Invariant to camera color matrix differences.
4. Spatial Random Crop & Jitter (85%-100% area, resized back to 224x224): Removing spatial alignment artifacts.
5. Temporal Consistency: Spatial cropping and color jitter parameters are shared across all 16 frames
   in a clip to preserve natural inter-frame motion coherence.
"""

import cv2
import numpy as np
import torch
import random
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class AugmentationConfig:
    enabled: bool = True
    jpeg_recompression_prob: float = 0.8
    jpeg_quality_range: Tuple[int, int] = (50, 95)
    gaussian_blur_prob: float = 0.3
    gaussian_kernel_sizes: Tuple[int, ...] = (3, 5)
    color_jitter_prob: float = 0.5
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    saturation_range: Tuple[float, float] = (0.85, 1.15)
    random_crop_prob: float = 0.5
    crop_scale_range: Tuple[float, float] = (0.85, 1.0)
    random_resample_prob: float = 0.4
    resample_scale_range: Tuple[float, float] = (0.6, 0.9)
    horizontal_flip_prob: float = 0.5


class AntiShortcutVideoAugmentor:
    """
    Applies temporally-coherent anti-shortcut augmentations to (16, 224, 224, 3) uint8 RGB frames.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def _apply_jpeg_recompression(self, frame: np.ndarray, quality: int) -> np.ndarray:
        """Simulates lossy compression via in-memory JPEG encode/decode."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        # cv2 expects BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode(".jpg", bgr, encode_param)
        if not result or encimg is None:
            return frame
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        if decimg is None:
            return frame
        return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    def _apply_gaussian_blur(self, frame: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
        """Applies gentle Gaussian blur to attenuate high-frequency container noise."""
        return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=sigma)

    def augment_clip(self, frames_array: np.ndarray) -> np.ndarray:
        """
        Augments a sequence of frames: input shape (T, H, W, 3) uint8 RGB.
        Returns augmented (T, H, W, 3) uint8 RGB array.
        """
        if not self.config.enabled:
            return frames_array

        T, H, W, C = frames_array.shape
        augmented = np.copy(frames_array)

        # 1. Temporally coherent Random Crop & Resize
        if random.random() < self.config.random_crop_prob:
            scale = random.uniform(*self.config.crop_scale_range)
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)

            cropped_frames = []
            for t in range(T):
                crop = augmented[t, top : top + crop_h, left : left + crop_w]
                resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
                cropped_frames.append(resized)
            augmented = np.stack(cropped_frames, axis=0)

        # 2. Temporally coherent Random Downsample & Resampling
        if random.random() < self.config.random_resample_prob:
            scale = random.uniform(*self.config.resample_scale_range)
            low_w = max(32, int(W * scale))
            low_h = max(32, int(H * scale))
            down_interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR])
            up_interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])

            resampled_frames = []
            for t in range(T):
                down = cv2.resize(augmented[t], (low_w, low_h), interpolation=down_interp)
                up = cv2.resize(down, (W, H), interpolation=up_interp)
                resampled_frames.append(up)
            augmented = np.stack(resampled_frames, axis=0)

        # 3. Temporally coherent Horizontal Flip
        if random.random() < self.config.horizontal_flip_prob:
            augmented = np.flip(augmented, axis=2).copy()

        # 4. Temporally coherent Color & Brightness Variation
        if random.random() < self.config.color_jitter_prob:
            b_factor = random.uniform(*self.config.brightness_range)
            c_factor = random.uniform(*self.config.contrast_range)
            s_factor = random.uniform(*self.config.saturation_range)

            jittered_frames = []
            for t in range(T):
                f = augmented[t].astype(np.float32)
                # Brightness
                f = f * b_factor
                # Contrast around mid-gray
                f = (f - 128.0) * c_factor + 128.0
                # Saturation in HSV
                f = np.clip(f, 0, 255).astype(np.uint8)
                hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
                f_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                jittered_frames.append(f_hsv)
            augmented = np.stack(jittered_frames, axis=0)

        # 4. Controlled JPEG Recompression
        if random.random() < self.config.jpeg_recompression_prob:
            quality = random.randint(*self.config.jpeg_quality_range)
            compressed_frames = []
            for t in range(T):
                c_frame = self._apply_jpeg_recompression(augmented[t], quality)
                compressed_frames.append(c_frame)
            augmented = np.stack(compressed_frames, axis=0)

        # 5. Gaussian Blur
        if random.random() < self.config.gaussian_blur_prob:
            ksize = random.choice(self.config.gaussian_kernel_sizes)
            sigma = random.uniform(0.3, 1.2)
            blurred_frames = []
            for t in range(T):
                b_frame = self._apply_gaussian_blur(augmented[t], ksize, sigma)
                blurred_frames.append(b_frame)
            augmented = np.stack(blurred_frames, axis=0)

        return augmented
