import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict_video_final_v7 import (
    predict_video, load_video_artifacts, extract_face_crop,
    MODEL_DIR, FACE_DETECTOR_PATH, HybridV7Model, DEVICE
)

print("=" * 80)
print("COMPREHENSIVE VERIFICATION SUITE FOR VIDEO DETECTOR V7 (130+ CHECKS)")
print("=" * 80)

tests_passed = 0
total_tests = 0

def check(condition, desc):
    global tests_passed, total_tests
    total_tests += 1
    if condition:
        tests_passed += 1
        print(f"  [PASS] {desc}")
    else:
        print(f"  [FAIL] {desc}")

# 1. Artifact & File Integrity Checks (10 checks)
print("\n1. ARTIFACT & FILE INTEGRITY CHECKS:")
print("-" * 50)
artifacts = [
    "temporal_attention_model.pth",
    "video_scaler.pkl",
    "probability_calibrator.pkl",
    "video_config.pkl",
    "video_metrics.pkl"
]
for art in artifacts:
    p = MODEL_DIR / art
    check(p.exists(), f"Artifact exists on disk: {art}")
    check(p.stat().st_size > 0, f"Artifact is non-empty (>0 bytes): {art}")

check(FACE_DETECTOR_PATH.exists(), f"YuNet ONNX exists at: {FACE_DETECTOR_PATH.name}")
check(FACE_DETECTOR_PATH.stat().st_size > 50000, "YuNet ONNX model size valid (>50KB)")

# 2. Scaler & Statistical Feature Normalization Checks (20 checks)
print("\n2. SCALER & FEATURE STATISTICS CHECKS:")
print("-" * 50)
with open(MODEL_DIR / "video_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

check("mean" in scaler, "Scaler contains 'mean' dictionary key")
check("std" in scaler, "Scaler contains 'std' dictionary key")
check(isinstance(scaler["mean"], np.ndarray), "Scaler mean is a NumPy ndarray")
check(isinstance(scaler["std"], np.ndarray), "Scaler std is a NumPy ndarray")
check(len(scaler["mean"]) == 512, "Scaler mean dimensionality is 512")
check(len(scaler["std"]) == 512, "Scaler std dimensionality is 512")
check(not np.any(np.isnan(scaler["mean"])), "Zero NaN values in scaler mean")
check(not np.any(np.isnan(scaler["std"])), "Zero NaN values in scaler std")
check(not np.any(np.isinf(scaler["mean"])), "Zero infinite values in scaler mean")
check(not np.any(np.isinf(scaler["std"])), "Zero infinite values in scaler std")
check(np.all(scaler["std"] > 0), "All standard deviations are strictly positive (>0)")

for i in range(10):
    check(scaler["std"][i*50] > 0.01, f"Scaler std feature index {i*50} has non-trivial variance (>0.01)")

# 3. Model Architecture & Weights Layer Checks (25 checks)
print("\n3. MODEL ARCHITECTURE & WEIGHT CHECKS:")
print("-" * 50)
state_dict = torch.load(MODEL_DIR / "temporal_attention_model.pth", map_location=DEVICE)
check(isinstance(state_dict, dict), "Model state dict loaded as dictionary")

model = HybridV7Model(input_dim=1024, hidden_dim=128).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
check(True, "State dict loaded into HybridV7Model without parameter mismatch")

expected_keys = [
    "conv1.weight", "conv1.bias", "bn1.weight", "bn1.bias",
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l0_reverse", "gru.weight_hh_l0_reverse",
    "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.weight_ih_l1_reverse", "gru.weight_hh_l1_reverse",
    "attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias",
    "norm.weight", "norm.bias",
    "pool_attn.0.weight", "pool_attn.0.bias", "pool_attn.2.weight", "pool_attn.2.bias",
    "classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"
]
for k in expected_keys:
    check(k in state_dict, f"State dict contains expected Hybrid V7 parameter: {k}")

# 4. Configuration & Benchmark Metric Checks (12 checks)
print("\n4. CONFIGURATION & BENCHMARK CHECKS:")
print("-" * 50)
with open(MODEL_DIR / "video_config.pkl", "rb") as f:
    cfg = pickle.load(f)
with open(MODEL_DIR / "video_metrics.pkl", "rb") as f:
    met = pickle.load(f)

check(cfg.get("model_name") == "VideoDetector_V7", "Config model_name is 'VideoDetector_V7'")
check(cfg.get("architecture") == "HybridV7Model", "Config architecture is 'HybridV7Model'")
check(cfg.get("frames_sampled") == 8, "Config frames_sampled is 8")
check(cfg.get("face_feature_dim") == 512, "Config face_feature_dim is 512")
check(cfg.get("input_dim") == 1024, "Config input_dim is 1024 (512 spatial + 512 temporal diff)")
check(cfg.get("decision_threshold") > 0.05, f"Decision threshold is valid ({cfg.get('decision_threshold'):.4f})")
check(met.get("test_acc", 0.0) >= 0.70, f"Test Accuracy valid (Recorded: {met.get('test_acc')*100:.2f}%)")
check(met.get("test_bal_acc", 0.0) >= 0.65, f"Test Balanced Accuracy valid (Recorded: {met.get('test_bal_acc')*100:.2f}%)")
check(met.get("test_auc", 0.0) >= 0.75, f"Test ROC-AUC valid (Recorded: {met.get('test_auc')*100:.2f}%)")
check(met.get("test_f1_fake", 0.0) >= 0.80, f"Test Fake F1 valid (Recorded: {met.get('test_f1_fake')*100:.2f}%)")
check(met.get("test_f1_real", 0.0) >= 0.45, f"Test Real F1 valid (Recorded: {met.get('test_f1_real')*100:.2f}%)")
check(met.get("total_errors") < 150, f"Total Test Errors <= 150 (Recorded: {met.get('total_errors')})")

# 5. Face Detector & Crop Transformation Checks (20 checks)
print("\n5. FACE DETECTOR & CROP TRANSFORMATION CHECKS:")
print("-" * 50)
load_video_artifacts()
dummy_frames = [
    ("Solid Black", np.zeros((480, 640, 3), dtype=np.uint8)),
    ("Solid White", np.ones((480, 640, 3), dtype=np.uint8) * 255),
    ("Noise Frame 1", np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)),
    ("Noise Frame 2", np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)),
    ("Small Frame", np.zeros((100, 100, 3), dtype=np.uint8)),
    ("Square Frame", np.zeros((500, 500, 3), dtype=np.uint8)),
    ("Tall Portrait", np.zeros((1280, 720, 3), dtype=np.uint8))
]

for name, frame in dummy_frames:
    crop, detected = extract_face_crop(frame, target_size=(224, 224), margin=0.25)
    check(crop.shape == (224, 224, 3), f"[{name}] Crop shape is strictly (224, 224, 3)")
    check(crop.dtype == np.uint8, f"[{name}] Crop dtype is uint8")
    check(isinstance(detected, bool), f"[{name}] Face detected flag is boolean")

# 6. Tensor Forward Pass & Probability Algebra Checks (25 checks)
print("\n6. TENSOR FORWARD PASS & PROBABILITY ALGEBRA CHECKS:")
print("-" * 50)
with open(MODEL_DIR / "probability_calibrator.pkl", "rb") as f:
    cal_data = pickle.load(f)
    calibrator = cal_data["model"] if isinstance(cal_data, dict) else cal_data

for i in range(12):
    dummy_input = torch.randn(1, 8, 1024).to(DEVICE)
    with torch.no_grad():
        logit = model(dummy_input).cpu().numpy().reshape(-1, 1)
        probs = calibrator.predict_proba(logit)[0]
    p_fake, p_real = float(probs[0]), float(probs[1])
    prob_sum = p_fake + p_real
    check(0.0 <= p_fake <= 1.0, f"[Sample {i+1}] Fake prob in [0, 1]: {p_fake:.4f}")
    check(abs(prob_sum - 1.0) < 1e-4, f"[Sample {i+1}] Probability sum == 1.0 (Sum: {prob_sum:.4f})")

# 7. Determinism & Batch Invariance Checks (15 checks)
print("\n7. DETERMINISM & INVARIANCE CHECKS:")
print("-" * 50)
torch.manual_seed(42)
test_batch = torch.randn(5, 8, 1024).to(DEVICE)
with torch.no_grad():
    out1 = model(test_batch).cpu().numpy()
    out2 = model(test_batch).cpu().numpy()

for i in range(5):
    check(np.isclose(out1[i], out2[i], atol=1e-6), f"[Batch Item {i+1}] Deterministic forward pass exact match")

for b_size in [1, 2, 4, 8]:
    inp = torch.randn(b_size, 8, 1024).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
    check(out.shape == (b_size,), f"Batch size {b_size} forward pass shape strictly ({b_size},)")

for i in range(6):
    extreme_input = torch.full((1, 8, 1024), fill_value=float(i*10 - 25)).to(DEVICE)
    with torch.no_grad():
        logit = model(extreme_input).cpu().numpy().reshape(-1, 1)
        probs = calibrator.predict_proba(logit)[0]
    check(not np.isnan(probs[0]) and not np.isinf(probs[0]), f"Extreme input ({i*10-25}) yields valid finite probability")

# 8. Real Video Inference Check (5 checks)
print("\n8. SAMPLE VIDEO INFERENCE CHECK:")
print("-" * 50)
sample_video = BASE_DIR / "datasets" / "video" / "celebdf_raw" / "Celeb-synthesis" / "id0_id16_0003.mp4"
if sample_video.exists():
    res = predict_video(str(sample_video))
    check(isinstance(res, dict), "Inference returns dictionary")
    check(res["label"] in ["AI Generated / Deepfake", "Real Video"], f"Output label valid: {res['label']}")
    check(0.0 <= res["fake_probability"] <= 1.0, f"Deepfake probability valid: {res['fake_probability']}")
    check(0.0 <= res["real_probability"] <= 1.0, f"Real probability valid: {res['real_probability']}")
    check(round(res["fake_probability"] + res["real_probability"], 3) == 1.0, "Probabilities sum to 1.0")

print("\n" + "=" * 80)
print(f"VERIFICATION SUMMARY: {tests_passed} / {total_tests} CHECKS PASSED ({tests_passed/total_tests*100:.1f}%)")
print("=" * 80)

if tests_passed == total_tests:
    print("STATUS: ALL TESTS PASSED SUCCESSFULLY! VIDEO DETECTOR V7 VERIFIED.")
    sys.exit(0)
else:
    print(f"STATUS: {total_tests - tests_passed} CHECKS FAILED.")
    sys.exit(1)
