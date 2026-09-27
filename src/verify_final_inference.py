import os
import sys
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict_text_final import predict_text
from src.predict_video_final import predict_video

print("=" * 80)
print("PHASE 3: COMPREHENSIVE BACKEND INFERENCE VERIFICATION")
print("=====================================================")

all_passed = True
total_tests = 0
passed_tests = 0

def check(condition, message):
    global all_passed, total_tests, passed_tests
    total_tests += 1
    if condition:
        passed_tests += 1
        print(f"  [PASS] {message}")
    else:
        all_passed = False
        print(f"  [FAIL] {message}")

# ==============================================================================
# 1. TEXT DETECTOR INFERENCE VERIFICATION
# ==============================================================================
print("\n" + "=" * 60)
print("1. TEXT DETECTOR INFERENCE TESTS")
print("=" * 60)

test_texts = [
    {
        "type": "Very Short Human",
        "text": "I think we should probably wait until tomorrow before deciding on that."
    },
    {
        "type": "Very Short AI",
        "text": "Furthermore, this phenomenon is deeply rooted in modern technological development."
    },
    {
        "type": "Medium Human",
        "text": "I've been working on this project for the past few weeks and honestly it has been really challenging to get everything running properly on Windows without crashing, but we're finally making good progress today."
    },
    {
        "type": "Medium AI",
        "text": "In conclusion, student-designed projects offer substantial advantages that enable individuals to succeed across various domains. Moreover, empirical research demonstrates that collaborative frameworks foster critical thinking and practical problem-solving capabilities."
    },
    {
        "type": "Long Human",
        "text": "Hey everyone, just wanted to give a quick update on what I found out during the team meeting yesterday afternoon. We discussed the budget allocations and decided that we need to cut down on extra travel expenses for next quarter. Let me know if anyone has questions about their specific department numbers before Friday's deadline so we can submit the final spreadsheet to management on time."
    },
    {
        "type": "Long AI",
        "text": "The advent of artificial intelligence has revolutionized numerous industries, fundamentally transforming traditional workflows and paradigms. Machine learning models, particularly large language models and neural architectures, demonstrate remarkable proficiency in natural language processing and semantic comprehension. Consequently, educational institutions and corporate enterprises are increasingly adapting their operational frameworks to harness these technological innovations effectively."
    },
    {
        "type": "Multilingual (German) AI",
        "text": "Der Erfinder der E-Mail ist am vergangenen Wochenende im Alter von 74 Jahren gestorben. Er entwickelte das grundlegende System bereits im Jahr 1971 und prägte damit die moderne digitale Kommunikation maßgeblich."
    },
    {
        "type": "Python Code Sample",
        "text": "def calculate_statistics(numbers):\n    if not numbers:\n        return None\n    mean_val = sum(numbers) / len(numbers)\n    return {'mean': mean_val, 'count': len(numbers)}"
    }
]

for item in test_texts:
    ttype = item["type"]
    text = item["text"]
    res = predict_text(text)
    
    ai_p = res["ai_probability"]
    hu_p = res["human_probability"]
    prob_sum = ai_p + hu_p
    sum_valid = abs(prob_sum - 1.0) < 0.0001
    
    print(f"\n--- Test: {ttype} ---")
    print(f"Input Snippet : {repr(text[:65])}...")
    print(f"Predicted     : {res['label']}")
    print(f"AI Prob       : {ai_p * 100:.2f}% | Human Prob: {hu_p * 100:.2f}% | Confidence: {res['confidence'] * 100:.2f}%")
    print(f"Length Tier   : {res['length_group']} | Applied Threshold: {res['threshold']:.2f}")
    
    check(res["label"] in ["AI Generated", "Human Written"], f"Label valid ('{res['label']}')")
    check(sum_valid, f"Probability sum strictly equals 100% (sum={prob_sum:.4f})")
    check(0.0 <= ai_p <= 1.0 and 0.0 <= hu_p <= 1.0, "Probabilities in valid range [0, 1]")
    check(res["confidence"] == max(ai_p, hu_p) if res["label"] == ("AI Generated" if ai_p >= res["threshold"] else "Human Written") else True, "Confidence matches predicted class probability")

# Edge Cases for Text
print("\n--- Text Edge Cases ---")
empty_res = predict_text("")
check(empty_res["label"] == "Human Written" and empty_res["human_probability"] == 1.0, "Empty string handled safely")

single_word_res = predict_text("Hello")
check(abs(single_word_res["ai_probability"] + single_word_res["human_probability"] - 1.0) < 0.0001, "Single word handled with valid probability sum")

# ==============================================================================
# 2. VIDEO DETECTOR INFERENCE VERIFICATION
# ==============================================================================
print("\n" + "=" * 60)
print("2. VIDEO DETECTOR INFERENCE TESTS")
print("=" * 60)

test_videos = [
    {
        "type": "Known Fake Deepfake Video",
        "path": BASE_DIR / "datasets" / "video" / "prepared" / "test" / "fake" / "id0_id16_0003.mp4"
    },
    {
        "type": "Known Real Video 1",
        "path": BASE_DIR / "datasets" / "video" / "prepared" / "test" / "real" / "00011.mp4"
    },
    {
        "type": "Known Real Video 2",
        "path": BASE_DIR / "datasets" / "video" / "prepared" / "test" / "real" / "00021.mp4"
    }
]

for vitem in test_videos:
    vtype = vitem["type"]
    vpath = vitem["path"]
    
    print(f"\n--- Test: {vtype} ---")
    print(f"Video Path    : {vpath.name}")
    
    if not vpath.exists():
        check(False, f"Test video file exists: {vpath}")
        continue
        
    vres = predict_video(str(vpath))
    fake_p = vres["fake_probability"]
    real_p = vres["real_probability"]
    vprob_sum = fake_p + real_p
    vsum_valid = abs(vprob_sum - 1.0) < 0.0001
    
    print(f"Predicted     : {vres['label']}")
    print(f"Deepfake Prob : {fake_p * 100:.2f}% | Real Prob: {real_p * 100:.2f}% | Confidence: {vres['confidence'] * 100:.2f}%")
    print(f"Threshold     : {vres['threshold']:.2f} | Faces Detected: {vres['face_detected_frames']}/{vres['frames_analyzed']}")
    
    check(vres["label"] in ["Real / Authentic", "AI Generated / Deepfake"], f"Video label valid ('{vres['label']}')")
    check(vsum_valid, f"Video probability sum strictly equals 100% (sum={vprob_sum:.4f})")
    check(0.0 <= fake_p <= 1.0 and 0.0 <= real_p <= 1.0, "Probabilities in valid range [0, 1]")
    check(vres["frames_analyzed"] == 8, "Exactly 8 frames analyzed")

# ==============================================================================
# 3. MODEL ARTIFACT AND ARCHITECTURE CONSISTENCY CHECK
# ==============================================================================
print("\n" + "=" * 60)
print("3. MODEL ARTIFACTS & ARCHITECTURE CONSISTENCY AUDIT")
print("=" * 60)

# Check Text Artifacts
text_dir = BASE_DIR / "models" / "text_final"
required_text_files = [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "multi_char_tfidf_vectorizer.pkl",
    "linguistic_scaler.pkl",
    "text_classifier.pkl",
    "probability_calibrator.pkl",
    "text_config.pkl",
    "text_metrics.pkl"
]
for tf in required_text_files:
    check((text_dir / tf).exists(), f"Text artifact exists: {tf}")

# Check Video Artifacts
video_dir = BASE_DIR / "models" / "video_final"
required_video_files = [
    "temporal_attention_model.pth",
    "video_scaler.pkl",
    "probability_calibrator.pkl",
    "video_config.pkl",
    "video_metrics.pkl"
]
for vf in required_video_files:
    check((video_dir / vf).exists(), f"Video artifact exists: {vf}")

check((BASE_DIR / "models" / "face_detector" / "face_detection_yunet.onnx").exists(), "YuNet Face Detector ONNX model exists")

# ==============================================================================
# 4. OBSOLETE MODEL DIRECTORY AUDIT
# ==============================================================================
print("\n" + "=" * 60)
print("4. OBSOLETE MODEL REPOSITORY AUDIT")
print("=" * 60)

model_dirs = [d for d in os.listdir(BASE_DIR / "models") if os.path.isdir(BASE_DIR / "models" / d)]
for md in sorted(model_dirs):
    if md in ["text_final", "video_final", "face_detector"]:
        status = "ACTIVE FINAL PRODUCTION MODEL"
    elif md.endswith("_backup"):
        status = "SAFE BACKUP CHECKPOINT"
    else:
        status = "HISTORICAL / EXPERIMENTAL CHECKPOINT (Safe to keep)"
    print(f"  Directory: {md:30s} | Status: {status}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print(f"VERIFICATION SUMMARY: {passed_tests} / {total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
if all_passed:
    print("ALL BACKEND VERIFICATION CHECKS PASSED PERFECTLY!")
else:
    print("SOME CHECKS FAILED! Please inspect output above.")
print("=" * 80)

if not all_passed:
    sys.exit(1)
