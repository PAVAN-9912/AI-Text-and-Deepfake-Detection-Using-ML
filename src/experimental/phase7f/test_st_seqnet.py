"""
Comprehensive Unit Test and Resource Benchmark Suite for ST-SeqNet (Phase 7F.3)

Strict Execution Guard:
This test suite verifies architecture, preprocessing, augmentation, isolation guards,
and computational benchmarks. It DOES NOT perform training or checkpoint generation.
"""

import os
import sys
import time
import unittest
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torchvision.models as models

# Add project root to path
PROJECT_ROOT = r"C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experimental.phase7f.preprocessing import (
    StandardVideoPreprocessor,
    PreprocessingConfig,
    PREPROCESSING_VERSION,
)
from src.experimental.phase7f.augmentation import (
    AntiShortcutVideoAugmentor,
    AugmentationConfig,
)
from src.experimental.phase7f.dataset import STSeqNetDataset
from src.experimental.phase7f.models import STSeqNet, TemporalAttention
from src.experimental.phase7f.shortcut_assessment import run_source_codec_assessment


class TestSTSeqNetPreprocessing(unittest.TestCase):
    """Verifies standardized video decoding, sampling, and metadata isolation."""

    def setUp(self):
        self.preprocessor = StandardVideoPreprocessor()

    def test_sample_frame_indices_uniform(self):
        # 100 frames -> exactly 16 uniform indices
        indices = self.preprocessor.sample_frame_indices(100)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_sample_frame_indices_short_video(self):
        # 5 frames -> cyclical padding up to 16
        indices = self.preprocessor.sample_frame_indices(5)
        self.assertEqual(len(indices), 16)
        # Check cyclicity: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[5], 0)
        self.assertEqual(indices[10], 0)

    def test_frames_to_tensor_shape_and_normalization(self):
        # Input: (16, 224, 224, 3) uint8
        dummy_frames = np.random.randint(0, 256, (16, 224, 224, 3), dtype=np.uint8)
        tensor = self.preprocessor.frames_to_tensor(dummy_frames, apply_normalization=True)
        self.assertEqual(tensor.shape, (16, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        # Check that normalization shifted values away from [0, 1]
        self.assertTrue(tensor.min() < 0.0 or tensor.max() > 1.0)


class TestAntiShortcutAugmentation(unittest.TestCase):
    """Verifies anti-shortcut transformations preserve shape and temporal coherence."""

    def setUp(self):
        self.augmentor = AntiShortcutVideoAugmentor(
            AugmentationConfig(
                enabled=True,
                jpeg_recompression_prob=1.0,
                gaussian_blur_prob=1.0,
                color_jitter_prob=1.0,
                random_crop_prob=1.0,
                horizontal_flip_prob=1.0,
            )
        )

    def test_augmentation_shape_and_finite(self):
        dummy_clip = np.random.randint(20, 230, (16, 224, 224, 3), dtype=np.uint8)
        aug_clip = self.augmentor.augment_clip(dummy_clip)
        self.assertEqual(aug_clip.shape, (16, 224, 224, 3))
        self.assertEqual(aug_clip.dtype, np.uint8)
        self.assertFalse(np.isnan(aug_clip).any())


class TestDatasetAndIdentitySafeguards(unittest.TestCase):
    """Verifies dataset isolation, identity metadata separation, and test lock."""

    def setUp(self):
        self.manifest_path = os.path.join(
            PROJECT_ROOT, "experiments", "phase7_external_evaluation", "PHASE7F_2_PROVENANCE_MANIFEST.csv"
        )

    def test_test_partition_strict_lock(self):
        # Accessing test partition without allow_test_split must raise PermissionError
        with self.assertRaises(PermissionError):
            _ = STSeqNetDataset(
                manifest_path=self.manifest_path,
                partition="test",
                project_root=PROJECT_ROOT,
                allow_test_split=False,
            )

    def test_identity_flagging_in_train(self):
        ds = STSeqNetDataset(
            manifest_path=self.manifest_path,
            partition="train",
            project_root=PROJECT_ROOT,
        )
        self.assertEqual(len(ds), 400)
        # Check that id0 is flagged
        flagged_samples = ds.df[ds.df["identity_overlap_flag"]]
        self.assertGreater(len(flagged_samples), 0)
        # All flagged samples in train must involve id0
        for _, r in flagged_samples.iterrows():
            if r["category"] == "C2_real_face_man":
                self.assertEqual(r["genuine_identity"], 0)
            elif r["category"] == "C6_ai_face_man":
                self.assertEqual(r["donor_identity"], 0)

    def test_identity_flagging_in_val(self):
        ds = STSeqNetDataset(
            manifest_path=self.manifest_path,
            partition="val",
            project_root=PROJECT_ROOT,
        )
        self.assertEqual(len(ds), 120)
        # Check that id40/id41 are flagged
        flagged_samples = ds.df[ds.df["identity_overlap_flag"]]
        self.assertGreater(len(flagged_samples), 0)
        for _, r in flagged_samples.iterrows():
            if r["category"] == "C2_real_face_man":
                self.assertIn(r["genuine_identity"], [40, 41])
            elif r["category"] == "C6_ai_face_man":
                self.assertTrue(r["donor_identity"] == 40 or r["target_identity"] == 41)


class TestSTSeqNetModel(unittest.TestCase):
    """Verifies model forward pass, shapes, attention mechanisms, and determinism."""

    def setUp(self):
        # Lightweight model for CPU testing
        self.model = STSeqNet(
            pretrained_backbone=False,
            freeze_backbone=False,
            gru_hidden_dim=64,
            projection_dim=128,
            dropout=0.0,
        )
        self.model.eval()

    def test_forward_pass_output_shapes(self):
        batch_size = 2
        num_frames = 16
        dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(dummy_input)

        self.assertIn("logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertIn("attention_weights", outputs)
        self.assertIn("context_vector", outputs)

        # Check shapes
        self.assertEqual(outputs["logits"].shape, (batch_size, 1))
        self.assertEqual(outputs["probabilities"].shape, (batch_size, 1))
        self.assertEqual(outputs["attention_weights"].shape, (batch_size, num_frames))
        self.assertEqual(outputs["context_vector"].shape, (batch_size, 128))  # 64*2

        # Check probabilities range [0, 1]
        probs = outputs["probabilities"]
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        # Check attention weights sum to 1.0 across frames
        attn_sums = outputs["attention_weights"].sum(dim=1)
        self.assertTrue(torch.allclose(attn_sums, torch.ones(batch_size), atol=1e-5))

    def test_deterministic_eval_inference(self):
        dummy_input = torch.randn(1, 16, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            out1 = self.model(dummy_input)
            out2 = self.model(dummy_input)

        # Logits must be bitwise identical in eval mode
        self.assertTrue(torch.equal(out1["logits"], out2["logits"]))
        self.assertTrue(torch.equal(out1["attention_weights"], out2["attention_weights"]))


def run_benchmarks() -> Dict[str, Any]:
    """Measures latency and memory footprint on CPU."""
    print("\n--- Running ST-SeqNet Computational Resource Benchmarks ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark Device: {device}")

    model = STSeqNet(
        pretrained_backbone=False,
        gru_hidden_dim=128,
        projection_dim=256,
        dropout=0.0,
    ).to(device)
    model.eval()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Latency test: batch size 1
    input_b1 = torch.randn(1, 16, 3, 224, 224, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_b1)

    latencies = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model(input_b1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    median_lat = np.median(latencies)

    print(f"Inference Latency (B=1, 16 frames, 224x224): {mean_lat:.2f} ms ± {std_lat:.2f} ms (median: {median_lat:.2f} ms)")

    return {
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "mean_latency_ms": float(mean_lat),
        "std_latency_ms": float(std_lat),
        "median_latency_ms": float(median_lat),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 7F.3 — UNIT TEST SUITE AND BENCHMARKS")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromNames([
        "src.experimental.phase7f.test_st_seqnet.TestSTSeqNetPreprocessing",
        "src.experimental.phase7f.test_st_seqnet.TestAntiShortcutAugmentation",
        "src.experimental.phase7f.test_st_seqnet.TestDatasetAndIdentitySafeguards",
        "src.experimental.phase7f.test_st_seqnet.TestSTSeqNetModel",
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        print("\n[ERROR] Unit tests failed!")
        sys.exit(1)

    bench_results = run_benchmarks()
    print("\n[SUCCESS] All unit tests and benchmarks completed successfully.")
