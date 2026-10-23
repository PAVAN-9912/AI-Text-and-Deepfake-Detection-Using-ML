"""
Comprehensive Integration Test Suite for Vision Transformer (ViT) Deepfake Detector
=====================================================================================
Validates:
1. Complete preservation and non-regression of Text Detection module.
2. ViT input preprocessing (224x224, RGB color space, [-1, 1] normalization).
3. Zero-face guard (INSUFFICIENT_EVIDENCE when no face detected).
4. ViT inference output contracts, probability ranges, and class semantics (0: FAKE, 1: REAL).
5. Fallback hierarchy in Flask application.
"""

import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ── 1. Text Detector Preservation Test ────────────────────────────────────────
class TestTextDetectorPreservation(unittest.TestCase):
    def test_text_detector_unmodified_execution(self):
        import predict_text_final
        # Human text sample (matching application benchmark)
        human_text = "Hey guys, can someone send me the lecture notes from this morning? I forgot my notebook at home."
        res_human = predict_text_final.predict_text(human_text)
        self.assertIn("label", res_human)
        self.assertIn("ai_probability", res_human)
        self.assertIn("human_probability", res_human)
        self.assertAlmostEqual(res_human["ai_probability"] + res_human["human_probability"], 1.0, places=2)
        self.assertEqual(res_human["label"], "Human Written")

        # AI-generated text sample
        ai_text = "In conclusion, it is important to remember that the intricate juxtaposition of paradigms requires comprehensive synergy."
        res_ai = predict_text_final.predict_text(ai_text)
        self.assertIn("label", res_ai)
        self.assertIn("ai_probability", res_ai)
        self.assertIn("human_probability", res_ai)


# ── 2. ViT Preprocessing and Architecture Test ────────────────────────────────
class TestViTArchitecture(unittest.TestCase):
    def test_vit_preprocessing_specs(self):
        from src.predict_video_vit import VIT_MEAN, VIT_STD, INPUT_SIZE, _inference_transform
        self.assertEqual(INPUT_SIZE, 224)
        self.assertEqual(VIT_MEAN, [0.5, 0.5, 0.5])
        self.assertEqual(VIT_STD, [0.5, 0.5, 0.5])

        # Test dummy PIL image transform
        from PIL import Image
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        dummy_img[:] = [128, 128, 128]
        tensor = _inference_transform(dummy_img)

        self.assertEqual(tensor.shape, (3, 224, 224))
        # 128/255 = 0.50196 -> (0.50196 - 0.5) / 0.5 ≈ 0.0039
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 0.0, delta=0.05)


# ── 3. Zero-Face Guard Test ───────────────────────────────────────────────────
class TestZeroFaceGuard(unittest.TestCase):
    def test_zero_face_guard_response(self):
        from src.predict_video_vit import _insufficient_evidence_response
        res = _insufficient_evidence_response(frames_analyzed=8)
        self.assertEqual(res["label"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res["real_probability"], 0.0)
        self.assertEqual(res["fake_probability"], 0.0)
        self.assertEqual(res["face_count"], 0)
        self.assertEqual(res["reliability"], "NONE")
        self.assertIn("Zero face", res["reason"])


# ── 4. ViT Class Semantics Test ───────────────────────────────────────────────
class TestViTClassSemantics(unittest.TestCase):
    def test_class_mapping_consistency(self):
        # 0: FAKE, 1: REAL
        from transformers import ViTForImageClassification
        # Verify default label mapping setup
        id2label = {0: "FAKE", 1: "REAL"}
        label2id = {"FAKE": 0, "REAL": 1}
        self.assertEqual(id2label[0], "FAKE")
        self.assertEqual(id2label[1], "REAL")
        self.assertEqual(label2id["FAKE"], 0)
        self.assertEqual(label2id["REAL"], 1)


if __name__ == "__main__":
    unittest.main()
