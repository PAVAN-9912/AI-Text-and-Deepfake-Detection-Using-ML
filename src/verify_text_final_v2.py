import os
import sys
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from predict_text_final_v2 import predict_text, MODEL_DIR

print("=" * 80)
print("PHASE I — COMPREHENSIVE VERIFICATION SUITE FOR TEXT DETECTOR V2")
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

# 1. Artifact Existence Checks
print("\n1. ARTIFACT EXISTENCE CHECKS:")
print("-" * 50)
artifacts = [
    "word_tfidf_vectorizer.pkl",
    "char_tfidf_vectorizer.pkl",
    "multi_char_tfidf_vectorizer.pkl",
    "linguistic_scaler.pkl",
    "text_classifier.pkl",
    "probability_calibrator.pkl",
    "text_config.pkl",
    "text_metrics.pkl"
]

for art in artifacts:
    p = os.path.join(MODEL_DIR, art)
    check(os.path.exists(p) and os.path.getsize(p) > 0, f"Artifact exists & non-empty: {art}")

# 2. Config & Schema Checks
print("\n2. CONFIGURATION & SCHEMA CHECKS:")
print("-" * 50)
with open(os.path.join(MODEL_DIR, "text_config.pkl"), "rb") as f:
    cfg = pickle.load(f)
with open(os.path.join(MODEL_DIR, "text_metrics.pkl"), "rb") as f:
    met = pickle.load(f)

check(cfg.get("model_version") == "TextDetector_V2", "Config model_version is 'TextDetector_V2'")
check(cfg.get("word_max_features") == 150000, "Config word_max_features is 150,000")
check(cfg.get("char_max_features") == 150000, "Config char_max_features is 150,000")
check(cfg.get("multi_char_max_features") == 5000, "Config multi_char_max_features is 5,000")
check(len(cfg.get("all_feature_names", [])) == 34, "Config all_feature_names has 34 features")
check(isinstance(cfg.get("length_thresholds"), dict), "Config length_thresholds is dict")
check(met.get("accuracy", 0.0) > 0.85, f"Metrics accuracy > 85% (Recorded: {met.get('accuracy')*100:.2f}%)")
check(met.get("roc_auc", 0.0) > 0.90, f"Metrics ROC-AUC > 90% (Recorded: {met.get('roc_auc')*100:.2f}%)")

# 3. Functional Inference Checks Across Lengths & Types
print("\n3. FUNCTIONAL INFERENCE & PROBABILITY MATHEMATICS CHECKS:")
print("-" * 50)

test_cases = [
    ("Empty String", ""),
    ("Whitespace", "    \n\t   "),
    ("Single Word", "Artificial"),
    ("Very Short", "This is a quick summary of the recent election results."),
    ("Short Human", "I spent yesterday evening walking around the lake with my dog and taking photos of the sunset."),
    ("Short AI", "In conclusion, artificial intelligence continues to transform modern healthcare with predictive analytics."),
    ("Medium Text", "Machine learning algorithms have evolved rapidly over the past decade. By utilizing deep neural networks, systems can now transcribe audio, translate languages, and generate photo-realistic images with unprecedented accuracy."),
    ("Long Text", "The concept of sustainable urban development has garnered widespread attention as cities face unprecedented population growth and environmental challenges. Integrating renewable energy grids, optimizing public transit networks, and preserving green spaces are essential strategies for reducing the carbon footprint of metropolitan areas. Furthermore, civic policies that incentivize energy-efficient architecture ensure long-term resilience against climatic volatility."),
    ("Unicode / Emoji", "This is fantastic news! 😊🚀 Super excited to see what happens next with the new release."),
    ("Multilingual / German", "Der neue Bericht über den Klimawandel zeigt deutliche Auswirkungen auf die Landwirtschaft in Europa."),
    ("Code Sample", "def calculate_statistics(numbers):\n    if not numbers:\n        return {'mean': 0.0, 'std': 0.0}\n    return {'mean': sum(numbers)/len(numbers)}")
]

for label, txt in test_cases:
    res = predict_text(txt)
    check(isinstance(res, dict), f"[{label}] Output is dictionary")
    check(res.get("model") == "TextDetector_V2", f"[{label}] Model field is 'TextDetector_V2'")
    check(res.get("label") in ["AI Generated", "Human Written"], f"[{label}] Valid label: {res.get('label')}")
    
    p_ai = res.get("ai_probability")
    p_hum = res.get("human_probability")
    conf = res.get("confidence")
    
    check(0.0 <= p_ai <= 1.0, f"[{label}] AI probability in [0, 1]: {p_ai}")
    check(0.0 <= p_hum <= 1.0, f"[{label}] Human probability in [0, 1]: {p_hum}")
    check(abs((p_ai + p_hum) - 1.0) < 1e-4, f"[{label}] Probability sum == 1.0 (Sum: {p_ai + p_hum:.4f})")
    check(not (np.isnan(p_ai) or np.isnan(p_hum)), f"[{label}] No NaN in probabilities")
    check(conf == p_ai or conf == p_hum, f"[{label}] Confidence matches predicted class probability")

# 4. Repeated Inference Consistency Check
print("\n4. REPEATED INFERENCE CONSISTENCY CHECK:")
print("-" * 50)
sample_text = "The quick brown fox jumps over the lazy dog in the sunny afternoon."
res1 = predict_text(sample_text)
res2 = predict_text(sample_text)
check(res1 == res2, "Identical input produces perfectly deterministic output across repeated runs")

print("\n" + "=" * 80)
print(f"VERIFICATION SUMMARY: {tests_passed} / {total_tests} checks passed ({(tests_passed/total_tests)*100:.1f}%)")
if tests_passed == total_tests:
    print("ALL TEXT DETECTOR V2 AUTOMATED CHECKS PASSED WITH 100% INTEGRITY!")
else:
    print(f"WARNING: {total_tests - tests_passed} checks failed!")
print("=" * 80)
