import os
import sys
import io
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from predict_text_final import predict_text
try:
    from src.predict_video_fusion import predict_video_fusion as active_predict_video
except Exception:
    from src.predict_video_vit import predict_video_vit as active_predict_video

print("=" * 80)
print("FLASK BACKEND SYSTEM INTEGRATION TEST SUITE")
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

client = app.test_client()

# ============================================================
# 1. HEALTH ENDPOINT TESTS
# ============================================================
print("\n1. HEALTH ENDPOINT (/api/health):")
print("-" * 50)
res = client.get("/api/health")
check(res.status_code == 200, "GET /api/health returns HTTP 200")
data = res.get_json()
check(data.get("status") == "healthy", f"Health status is 'healthy' (Got: {data.get('status')})")
check(data.get("text_model") is True, "text_model flag is True")
check(data.get("video_model") is True, "video_model flag is True")
check(data.get("minilm") is True, "minilm flag is True")
check(data.get("face_detector") is True, "face_detector flag is True")
check(data.get("video_engine") == "PolicyC_STSeqNet_FaceViT", f"video_engine is PolicyC_STSeqNet_FaceViT (Got: {data.get('video_engine')})")
check(data.get("feature_count") == 194897, f"Feature count matches Text V3.4 (194,897): {data.get('feature_count')}")

# ============================================================
# 2. METRICS ENDPOINT TESTS
# ============================================================
print("\n2. METRICS ENDPOINT (/api/metrics):")
print("-" * 50)
res = client.get("/api/metrics")
check(res.status_code == 200, "GET /api/metrics returns HTTP 200")
data = res.get_json()
check(data.get("success") is True, "Metrics response success is True")
text_m = data.get("text_metrics", {})
check(text_m.get("accuracy") == 89.88, f"Text accuracy is 89.88% (Got: {text_m.get('accuracy')}%)")
check(text_m.get("roc_auc") == 95.63, f"Text ROC-AUC is 95.63% (Got: {text_m.get('roc_auc')}%)")
check(text_m.get("f1") == 89.62, f"Text F1 is 89.62% (Got: {text_m.get('f1')}%)")

video_m = data.get("video_metrics", {})
check("accuracy" in video_m and video_m.get("accuracy") > 70.0, f"Video accuracy is valid: {video_m.get('accuracy')}%")
check("roc_auc" in video_m and video_m.get("roc_auc") > 80.0, f"Video ROC-AUC is valid: {video_m.get('roc_auc')}%")
check("f1" in video_m and video_m.get("f1") > 70.0, f"Video F1 is valid: {video_m.get('f1')}%")
check("model" in video_m, f"Active video model is reported: {video_m.get('model')}")
check("threshold" in video_m, f"Active threshold is reported: {video_m.get('threshold')}")

# ============================================================
# 3. TEXT ANALYSIS API TESTS
# ============================================================
print("\n3. TEXT ANALYSIS API (/api/analyze/text):")
print("-" * 50)

# 3.1 Normal AI Text
ai_sample = (
    "The integration of deep learning and natural language processing has fundamentally "
    "transformed artificial content synthesis. Furthermore, multi-head attention mechanisms "
    "enable transformer architectures to model long-range semantic dependencies with unprecedented accuracy."
)
res = client.post("/api/analyze/text", json={"text": ai_sample})
check(res.status_code == 200, "POST /api/analyze/text (AI text) returns HTTP 200")
data = res.get_json()
check(data.get("success") is True, "Analysis success is True")
check(data.get("result") in ["AI-GENERATED", "HUMAN-WRITTEN"], f"Result badge is valid: {data.get('result')}")
check(data.get("ai_probability") is not None and data.get("human_probability") is not None, "AI & Human probabilities present")
prob_sum = data.get("ai_probability") + data.get("human_probability")
check(abs(prob_sum - 100.0) < 0.05, f"AI + Human probabilities sum to 100.0% (Sum: {prob_sum}%)")
check(data.get("confidence") == max(data.get("ai_probability"), data.get("human_probability")), "Confidence matches predicted class probability")
check("linguistic_features" in data, "Linguistic features dictionary present in payload")
check("log_word_count" in data["linguistic_features"], "log_word_count present in linguistic_features")

# 3.2 Normal Human Text
human_sample = (
    "Hey everyone, just wanted to give a quick update on the project. I left my laptop charger "
    "at the library yesterday so I couldn't finish the slides, but I'll get them to you guys by noon today!"
)
res = client.post("/api/analyze/text", json={"text": human_sample})
check(res.status_code == 200, "POST /api/analyze/text (Human text) returns HTTP 200")
data = res.get_json()
check(data.get("success") is True, "Human sample analysis success is True")
prob_sum = data.get("ai_probability") + data.get("human_probability")
check(abs(prob_sum - 100.0) < 0.05, f"Human sample probability sum == 100.0% (Sum: {prob_sum}%)")

# 3.3 Multilingual (German) Text
german_sample = (
    "Der Erfinder der E-Mail ist am vergangenen Wochenende im Alter von 74 Jahren gestorben. "
    "Ray Tomlinson führte das @-Zeichen als Trennung zwischen Benutzer und Rechnername ein."
)
res = client.post("/api/analyze/text", json={"text": german_sample})
check(res.status_code == 200, "POST /api/analyze/text (German text) returns HTTP 200")
data = res.get_json()
check(data.get("success") is True, "German text analysis succeeds")

# 3.4 Python Code Sample
code_sample = """def compute_loss(predictions, targets):
    if not predictions:
        return 0.0
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)"""
res = client.post("/api/analyze/text", json={"text": code_sample})
check(res.status_code == 200, "POST /api/analyze/text (Code sample) returns HTTP 200")
data = res.get_json()
check(data.get("success") is True, "Code sample analysis succeeds")

# 3.5 Very Short Text (1-2 words -> INSUFFICIENT_EVIDENCE)
res = client.post("/api/analyze/text", json={"text": "Hello world"})
check(res.status_code == 200, "POST /api/analyze/text (2 words) returns HTTP 200")
data = res.get_json()
check(data.get("result") == "INSUFFICIENT_EVIDENCE", f"2 words returns INSUFFICIENT_EVIDENCE (Got: {data.get('result')})")
check(data.get("is_short_text") is True, "is_short_text flag is True")
check(data.get("confidence") is None, "Confidence is None for insufficient evidence")

# 3.6 Error Cases: Empty and Missing JSON
res = client.post("/api/analyze/text", json={"text": ""})
check(res.status_code == 400, "Empty text returns HTTP 400")
check(res.get_json().get("success") is False, "Empty text returns success=False")

res = client.post("/api/analyze/text", data="not json", content_type="text/plain")
check(res.status_code == 400, "Non-JSON payload returns HTTP 400")

# ============================================================
# 4. VIDEO ANALYSIS API TESTS
# ============================================================
print("\n4. VIDEO ANALYSIS API (/api/analyze/video):")
print("-" * 50)

# 4.1 Video Inference (Deepfake Sample)
sample_video_fake = BASE_DIR / "datasets" / "video" / "celebdf_raw" / "Celeb-synthesis" / "id0_id16_0003.mp4"
if sample_video_fake.exists():
    with open(sample_video_fake, "rb") as vf:
        video_bytes = vf.read()
    
    data = {"file": (io.BytesIO(video_bytes), "test_fake_video.mp4")}
    res = client.post("/api/predict/video", data=data, content_type="multipart/form-data")
    check(res.status_code == 200, "POST /api/predict/video (valid fake video) returns HTTP 200")
    vdata = res.get_json()
    check(vdata.get("success") is True, "Video analysis success is True")
    check(vdata.get("result") in ["REAL", "FAKE"], f"Soft Binary Fusion always returns REAL or FAKE (Got: {vdata.get('result')})")
    check(vdata.get("display_verdict") in ["HUMAN", "AI-GENERATED"], f"display_verdict is HUMAN or AI-GENERATED (Got: {vdata.get('display_verdict')})")
    check("PolicyC_STSeqNet_FaceViT" in vdata.get("model", ""), f"Model name is PolicyC_STSeqNet_FaceViT (Got: {vdata.get('model')})")
    check(vdata.get("frames_analyzed") == 16, f"ST-SeqNet frames analyzed strictly 16 (Got: {vdata.get('frames_analyzed')})")
    check("st_seqnet_p_ai" in vdata and vdata.get("st_seqnet_p_ai") is not None, "st_seqnet_p_ai present in response")
    check("routing_rule" in vdata and vdata.get("routing_rule") is not None, f"routing_rule present: {vdata.get('routing_rule')}")
    check("attention_weights" in vdata and len(vdata.get("attention_weights", [])) == 16, "16 temporal attention_weights present")
    check("probabilities" in vdata, "probabilities dict present in response")
    if vdata.get("result") in ["REAL", "FAKE"]:
        check(vdata.get("real_probability") is not None and vdata.get("fake_probability") is not None, "Real & Fake probabilities present")
        v_prob_sum = vdata.get("real_probability") + vdata.get("fake_probability")
        check(abs(v_prob_sum - 100.0) < 0.05, f"Video Real + Fake probabilities sum to 100.0% (Sum: {v_prob_sum}%)")
        expected_conf = vdata["real_probability"] if vdata["result"] == "REAL" else vdata["fake_probability"]
        check(abs(vdata["confidence"] - expected_conf) < 1e-4, f"Confidence strictly matches predicted class: {vdata['confidence']}%")
    check(vdata.get("video_filename") == "test_fake_video.mp4", "Video filename preserved in response")

# 4.1b Verify all 4 Soft Binary Fusion branches (Cases A-D)
from src.predict_video_fusion import fuse_binary_probabilities

p_a, rule_a, _ = fuse_binary_probabilities(0.82, None, 0)
check(abs(p_a - 0.82) < 1e-6 and rule_a == "CASE_A_FACELESS_ST_SEQNET", f"Case A (No Faces): p_ai={p_a}, rule={rule_a}")

p_b, rule_b, _ = fuse_binary_probabilities(0.30, 0.70, 3)
check(abs(p_b - 0.70) < 1e-6 and rule_b == "CASE_B_STRONG_FACIAL_MANIPULATION", f"Case B (Strong FaceSwap): p_ai={p_b}, rule={rule_b}")

p_c, rule_c, _ = fuse_binary_probabilities(0.90, 0.10, 4)
expected_c = 0.35 * 0.90 + 0.65 * 0.10  # 0.38 < 0.50 -> REAL (HUMAN)!
check(abs(p_c - expected_c) < 1e-6 and p_c < 0.50 and rule_c == "CASE_C_AUTHENTIC_FACE_DAMPENING", f"Case C (Authentic Face Dampening): p_ai={p_c:.4f} (<0.50), rule={rule_c}")

p_d, rule_d, _ = fuse_binary_probabilities(0.80, 0.40, 2)
expected_d = 0.65 * 0.80 + 0.35 * 0.40  # 0.66 >= 0.50 -> FAKE (AI-GENERATED)!
check(abs(p_d - expected_d) < 1e-6 and p_d >= 0.50 and rule_d == "CASE_D_DUAL_EXPERT_BLEND", f"Case D (Dual-Expert Blend): p_ai={p_d:.4f} (>=0.50), rule={rule_d}")

# 4.1c Pillarbox 9:16 Border Cropping Verification
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf_pb:
    temp_pb = tf_pb.name
try:
    fourcc_pb = cv2.VideoWriter_fourcc(*"mp4v")
    out_pb = cv2.VideoWriter(temp_pb, fourcc_pb, 10, (320, 240))
    for _ in range(16):
        frame_pb = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_pb[:, 100:220, :] = 180
        out_pb.write(frame_pb)
    out_pb.release()
    pb_res = active_predict_video(temp_pb)
    check(pb_res.get("border_cropped") is True, "Pillarbox black bars (>15% area) stripped via border_cropped=True")
    check(pb_res.get("label") in ("REAL", "FAKE"), f"Pillarbox video produces binary verdict: {pb_res.get('label')}")
finally:
    if os.path.exists(temp_pb):
        os.remove(temp_pb)

# 4.2 Video Inference (Real Video Sample)
sample_video_real = BASE_DIR / "datasets" / "video" / "celebdf_raw" / "YouTube-real" / "00000.mp4"
if not sample_video_real.exists():
    sample_video_real = BASE_DIR / "datasets" / "video" / "celebdf_raw" / "Celeb-real" / "id0_0000.mp4"

if sample_video_real.exists():
    with open(sample_video_real, "rb") as vf:
        r_bytes = vf.read()
    data_r = {"file": (io.BytesIO(r_bytes), "test_real_video.mp4")}
    res_r = client.post("/api/analyze/video", data=data_r, content_type="multipart/form-data")
    check(res_r.status_code == 200, "POST /api/analyze/video (valid real video) returns HTTP 200")
    vr_data = res_r.get_json()
    check(vr_data.get("success") is True, "Real video analysis success is True")
    if vr_data.get("result") in ["REAL", "FAKE"]:
        vr_prob_sum = vr_data.get("real_probability") + vr_data.get("fake_probability")
        check(abs(vr_prob_sum - 100.0) < 0.05, f"Real video probabilities sum to 100.0%: {vr_prob_sum}%")
        expected_conf = vr_data["real_probability"] if vr_data.get("result") == "REAL" else vr_data["fake_probability"]
        check(abs(vr_data["confidence"] - expected_conf) < 1e-4, f"Confidence matches predicted class probability: {vr_data['confidence']}%")
    else:
        check(vr_data.get("result") == "INCONCLUSIVE", f"Valid inconclusive result: {vr_data.get('result')}")

# 4.3 Error Handling: Missing file
res = client.post("/api/analyze/video", data={}, content_type="multipart/form-data")
check(res.status_code == 400, "POST /api/analyze/video without file returns HTTP 400")

# 4.4 Error Handling: Invalid file extension
bad_file_data = {"file": (io.BytesIO(b"dummy content"), "document.txt")}
res = client.post("/api/analyze/video", data=bad_file_data, content_type="multipart/form-data")
check(res.status_code == 400, "POST /api/analyze/video with .txt file returns HTTP 400")
check("Unsupported video format" in res.get_json().get("error", ""), "Clear error message for invalid format")

# 4.5 Zero-face handling: Non-face video returns INSUFFICIENT_EVIDENCE
import tempfile
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
    temp_noface = tf.name
try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_noface, fourcc, 10, (224, 224))
    for _ in range(10):
        out.write(np.zeros((224, 224, 3), dtype=np.uint8))
    out.release()
    with open(temp_noface, "rb") as vf:
        noface_bytes = vf.read()
    res_nf = client.post("/api/analyze/video", data={"file": (io.BytesIO(noface_bytes), "noface.mp4")}, content_type="multipart/form-data")
    check(res_nf.status_code == 200, "POST /api/analyze/video (zero-face video) returns HTTP 200")
    nf_data = res_nf.get_json()
    check(nf_data.get("result") in ("REAL", "FAKE"), f"Zero-face video delegated via Policy C to ST-SeqNet (Got: {nf_data.get('result')})")
    check(nf_data.get("routing_rule") == "CASE_A_FACELESS_ST_SEQNET", f"Zero-face routing rule is CASE_A_FACELESS_ST_SEQNET (Got: {nf_data.get('routing_rule')})")
    check(nf_data.get("face_detected_frames") == 0, "Zero-face video reports face_detected_frames == 0")
    check(nf_data.get("accepted_crops") == 0, "Zero-face video reports accepted_crops == 0")
    check(nf_data.get("reliability") == "FULL_FRAME", f"Zero-face video reports reliability == 'FULL_FRAME' (Got: {nf_data.get('reliability')})")
    check(nf_data.get("confidence") is not None, "Zero-face video reports valid confidence")
finally:
    if os.path.exists(temp_noface):
        os.remove(temp_noface)

# ============================================================
# 5. CROSS-VALIDATION: STANDALONE VS FLASK API
# ============================================================
print("\n5. CROSS-VALIDATION: STANDALONE VS FLASK API INFERENCE:")
print("-" * 50)

# 5.1 Text Standalone vs API
test_text = (
    "Artificial intelligence is rapidly advancing across various technological disciplines. "
    "Researchers continue to develop state of the art deep neural networks."
)
standalone_text_res = predict_text(test_text)
api_text_res = client.post("/api/analyze/text", json={"text": test_text}).get_json()

standalone_ai_prob_pct = round(standalone_text_res["ai_probability"] * 100, 2)
api_ai_prob_pct = api_text_res["ai_probability"]

check(
    abs(standalone_ai_prob_pct - api_ai_prob_pct) < 1e-4,
    f"Text AI probability exact match: Standalone={standalone_ai_prob_pct}%, API={api_ai_prob_pct}%"
)
expected_label = "AI-GENERATED" if standalone_text_res["label"] == "AI Generated" else "HUMAN-WRITTEN"
check(
    api_text_res["result"] == expected_label,
    f"Text label exact match: Standalone={standalone_text_res['label']}, API={api_text_res['result']}"
)

# 5.2 Video Standalone vs API (Fake Video)
if sample_video_fake.exists():
    standalone_video_res = active_predict_video(str(sample_video_fake))
    with open(sample_video_fake, "rb") as vf:
        v_bytes = vf.read()
    api_video_res = client.post(
        "/api/analyze/video",
        data={"file": (io.BytesIO(v_bytes), "sample.mp4")},
        content_type="multipart/form-data"
    ).get_json()

    standalone_fake_pct = round(standalone_video_res["fake_probability"] * 100, 2)
    api_fake_pct = api_video_res["fake_probability"]

    check(
        abs(standalone_fake_pct - api_fake_pct) < 1e-4,
        f"Video Fake probability exact match: Standalone={standalone_fake_pct}%, API={api_fake_pct}%"
    )
    expected_v_label = standalone_video_res.get("label", "FAKE")
    check(
        api_video_res["result"] == expected_v_label,
        f"Video label exact match: Standalone={expected_v_label}, API={api_video_res['result']}"
    )

# 5.3 Video Standalone vs API (Real Video)
if sample_video_real.exists():
    standalone_real_res = active_predict_video(str(sample_video_real))
    with open(sample_video_real, "rb") as vf:
        vr_bytes = vf.read()
    api_real_res = client.post(
        "/api/analyze/video",
        data={"file": (io.BytesIO(vr_bytes), "sample_real.mp4")},
        content_type="multipart/form-data"
    ).get_json()

    expected_real_label = standalone_real_res.get("label", "REAL")
    check(
        api_real_res["result"] == expected_real_label,
        f"Real Video label exact match: Standalone={expected_real_label}, API={api_real_res['result']}"
    )
    if standalone_real_res.get("real_probability") is not None and api_real_res.get("real_probability") is not None:
        standalone_real_pct = round(standalone_real_res["real_probability"] * 100, 2)
        api_real_pct = api_real_res["real_probability"]
        check(
            abs(standalone_real_pct - api_real_pct) < 1e-4,
            f"Real Video probability exact match: Standalone={standalone_real_pct}%, API={api_real_pct}%"
        )
    else:
        check(
            abs((standalone_real_res.get("st_seqnet_p_ai") or 0.0) - (api_real_res.get("st_seqnet_p_ai") or 0.0)) < 1e-6,
            f"Real Video INCONCLUSIVE ST-SeqNet score exact match: {api_real_res.get('st_seqnet_p_ai')}"
        )

# ============================================================
# 6. ARCHITECTURE METADATA INTEGRITY CHECKS
# ============================================================
print("\n6. ARCHITECTURE METADATA INTEGRITY AUDIT:")
print("-" * 50)

html_file = BASE_DIR / "templates" / "index.html"
js_file = BASE_DIR / "static" / "js" / "app.js"

html_text = html_file.read_text(encoding="utf-8")
js_text = js_file.read_text(encoding="utf-8")

check("BiLSTM" not in html_text, "templates/index.html does not contain stale 'BiLSTM'")
check("BiLSTM" not in js_text, "static/js/app.js does not contain stale 'BiLSTM'")
check("Temporal Conv1D" in html_text, "templates/index.html contains 'Temporal Conv1D'")
check("BiGRU" in html_text, "templates/index.html contains 'BiGRU'")
check("Multi-Head Attention" in html_text, "templates/index.html contains 'Multi-Head Attention'")

print("\n" + "=" * 80)
print(f"INTEGRATION TEST SUMMARY: {tests_passed} / {total_tests} CHECKS PASSED ({tests_passed/total_tests*100:.1f}%)")
print("=" * 80)

if tests_passed == total_tests:
    print("STATUS: ALL FLASK INTEGRATION TESTS PASSED PERFECTLY!")
    sys.exit(0)
else:
    print(f"STATUS: {total_tests - tests_passed} INTEGRATION TESTS FAILED.")
    sys.exit(1)
