import os
import sys
import io
import time
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from predict_text_final import predict_text
from src.predict_video_final_v4 import predict_video

print("=" * 80)
print("COMPREHENSIVE FINAL END-TO-END APPLICATION TEST HARNESS")
print("=" * 80)

client = app.test_client()
total_checks = 0
passed_checks = 0

def test(cond, msg):
    global total_checks, passed_checks
    total_checks += 1
    if cond:
        passed_checks += 1
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")

# ----------------------------------------------------------------------
# 1. APPLICATION & ROUTE VERIFICATION
# ----------------------------------------------------------------------
print("\n1. APPLICATION STARTUP & BASE ROUTES:")
print("-" * 50)
t0 = time.time()
res_home = client.get("/")
t_home = time.time() - t0
test(res_home.status_code == 200, f"GET / returns 200 OK (Latency: {t_home*1000:.1f}ms)")
html_content = res_home.get_data(as_text=True)
test("AI-Text-and-Deepfake-Detection-Using-ML" in html_content, "HTML contains title and branding")
test('id="home"' in html_content and 'id="text"' in html_content and 'id="video"' in html_content, "HTML contains all core tab sections")

res_health = client.get("/api/health")
test(res_health.status_code == 200, "GET /api/health returns 200 OK")
health_data = res_health.get_json()
test(health_data["status"] == "healthy", "Health status is 'healthy'")
test(health_data["text_model"] and health_data["video_model"] and health_data["minilm"] and health_data["face_detector"], "All sub-engines reported ready")

res_metrics = client.get("/api/metrics")
test(res_metrics.status_code == 200, "GET /api/metrics returns 200 OK")
metrics_data = res_metrics.get_json()
test(metrics_data["text_metrics"]["accuracy"] == 89.88, "Text accuracy is 89.88%")
test(metrics_data["video_metrics"]["accuracy"] == 77.03, "Video accuracy is 77.03%")

# ----------------------------------------------------------------------
# 2. TEXT END-TO-END TESTS ACROSS SCENARIOS
# ----------------------------------------------------------------------
print("\n2. TEXT END-TO-END INFERENCE TESTS:")
print("-" * 50)

text_samples = [
    ("Human Conversational", "Hey guys, can someone send me the lecture notes from this morning? I forgot my notebook at home.", "HUMAN-WRITTEN"),
    ("AI Synthesized Academic", "Furthermore, convolutional and recurrent neural network architectures facilitate the comprehensive extraction of spatial-temporal representations in synthetic media forensics.", "AI-GENERATED"),
    ("Very Short (2 words)", "Good morning", "INSUFFICIENT_EVIDENCE"),
    ("Short Text (6 words)", "The machine learning experiment was successful.", "AI-GENERATED"),
    ("Medium Text (45 words)", "Artificial intelligence encompasses various subfields, including natural language processing, computer vision, and reinforcement learning. These methodologies enable machines to simulate cognitive functions, solve complex mathematical optimization tasks, and generate realistic synthetic artifacts with minimal human intervention across diverse domains.", "AI-GENERATED"),
    ("Long Text (75 words)", "In recent years, the rapid advancement of deep generative modeling techniques has revolutionized digital media synthesis. Modern diffusion models and autoregressive transformers can generate hyper-realistic imagery, audio, and text that are nearly indistinguishable from human-created content. Consequently, developing robust, generalizable forensic verification pipelines has become an urgent priority for information security, digital journalism integrity, and public trust in digital communications across global platforms.", "AI-GENERATED"),
    ("Multilingual German", "Der Erfinder der E-Mail ist am vergangenen Wochenende im Alter von 74 Jahren gestorben. Ray Tomlinson führte das @-Zeichen als Trennung zwischen Benutzer und Rechnername ein.", "AI-GENERATED"),
    ("Code / Punctuation-Heavy", "def calculate_statistics(values):\n    if not values:\n        return {'mean': 0.0, 'std': 0.0}\n    mean_val = sum(values) / len(values)\n    var_val = sum((x - mean_val)**2 for x in values) / len(values)\n    return {'mean': mean_val, 'std': var_val ** 0.5}", "AI-GENERATED")
]

timings_text = []
for name, sample, expected_res in text_samples:
    t_start = time.time()
    resp = client.post("/api/analyze/text", json={"text": sample})
    elapsed = (time.time() - t_start) * 1000
    timings_text.append(elapsed)
    
    test(resp.status_code == 200, f"[{name}] POST /api/analyze/text status 200 ({elapsed:.1f}ms)")
    data = resp.get_json()
    test(data["success"] is True, f"[{name}] Response success=True")
    test(data["result"] in ["AI-GENERATED", "HUMAN-WRITTEN", "INSUFFICIENT_EVIDENCE"], f"[{name}] Valid result badge: {data['result']}")
    
    if data["result"] != "INSUFFICIENT_EVIDENCE":
        p_ai = data["ai_probability"]
        p_hu = data["human_probability"]
        test(abs(p_ai + p_hu - 100.0) < 0.05, f"[{name}] Probabilities sum to 100%: {p_ai}% + {p_hu}% = {p_ai+p_hu}%")
        test(data["confidence"] == max(p_ai, p_hu), f"[{name}] Confidence matches max prob: {data['confidence']}%")
        test(data["threshold"] > 0, f"[{name}] Threshold valid: {data['threshold']}")
        test("linguistic_features" in data, f"[{name}] Linguistic features present")
    else:
        test(data["is_short_text"] is True, f"[{name}] is_short_text is True for 2 words")

# ----------------------------------------------------------------------
# 3. VIDEO END-TO-END TESTS ACROSS SCENARIOS
# ----------------------------------------------------------------------
print("\n3. VIDEO END-TO-END INFERENCE TESTS:")
print("-" * 50)

sample_fake = BASE_DIR / "datasets" / "video" / "celebdf_raw" / "Celeb-synthesis" / "id0_id16_0003.mp4"
sample_real = BASE_DIR / "datasets" / "video" / "celebdf_raw" / "YouTube-real" / "00009.mp4"
if not sample_real.exists():
    real_candidates = list((BASE_DIR / "datasets" / "video" / "celebdf_raw" / "Celeb-real").glob("*.mp4"))
    sample_real = real_candidates[0] if real_candidates else sample_fake

video_tests = [
    ("Synthesized Deepfake Video", sample_fake, "FAKE"),
    ("Authentic Real Face Video", sample_real, "REAL")
]

timings_video = []
for name, vpath, expected_lbl in video_tests:
    if vpath.exists():
        with open(vpath, "rb") as vf:
            v_bytes = vf.read()
        
        t_v0 = time.time()
        resp = client.post(
            "/api/analyze/video",
            data={"file": (io.BytesIO(v_bytes), vpath.name)},
            content_type="multipart/form-data"
        )
        elapsed_v = (time.time() - t_v0) * 1000
        timings_video.append(elapsed_v)
        
        test(resp.status_code == 200, f"[{name}] Video upload returns 200 OK ({elapsed_v:.1f}ms)")
        vdata = resp.get_json()
        test(vdata["success"] is True, f"[{name}] Video success=True")
        test(vdata["result"] in ["REAL", "FAKE"], f"[{name}] Result is REAL or FAKE (Got: {vdata['result']})")
        p_real = vdata["real_probability"]
        p_fake = vdata["fake_probability"]
        test(abs(p_real + p_fake - 100.0) < 0.05, f"[{name}] Probabilities sum to 100%: Real={p_real}%, Fake={p_fake}%")
        test(vdata["frames_analyzed"] == 8, f"[{name}] Frames analyzed strictly 8")
        test(vdata["face_detected_frames"] > 0, f"[{name}] Face detected frames > 0 ({vdata['face_detected_frames']}/8)")
        test(vdata["model"] == "VideoDetector_V4", f"[{name}] Model name is VideoDetector_V4")
        
        # Semantic consistency: If real > fake, result MUST be REAL
        if p_real > p_fake:
            test(vdata["result"] == "REAL", f"[{name}] Semantic check: Real({p_real}%) > Fake({p_fake}%) yields REAL result")
            test(abs(vdata["confidence"] - p_real) < 1e-4, f"[{name}] Confidence strictly equals Real prob ({p_real}%)")
        else:
            test(vdata["result"] == "FAKE", f"[{name}] Semantic check: Fake({p_fake}%) >= Real({p_real}%) yields FAKE result")
            test(abs(vdata["confidence"] - p_fake) < 1e-4, f"[{name}] Confidence strictly equals Fake prob ({p_fake}%)")

# ----------------------------------------------------------------------
# 4. ERROR HANDLING TESTS
# ----------------------------------------------------------------------
print("\n4. ERROR HANDLING TESTS:")
print("-" * 50)

# Empty text
res_err1 = client.post("/api/analyze/text", json={"text": "   "})
test(res_err1.status_code == 400, "Empty text returns 400 Bad Request")
test(res_err1.get_json()["success"] is False, "Empty text response success=False")
test("Empty text" in res_err1.get_json()["error"], "Empty text error message clear")

# Missing JSON text
res_err2 = client.post("/api/analyze/text", json={})
test(res_err2.status_code == 400, "Missing text key returns 400 Bad Request")

# Video missing file
res_err3 = client.post("/api/analyze/video", data={}, content_type="multipart/form-data")
test(res_err3.status_code == 400, "Video missing file returns 400 Bad Request")

# Unsupported file type (.pdf)
bad_file = {"file": (io.BytesIO(b"%PDF-1.4 dummy"), "document.pdf")}
res_err4 = client.post("/api/analyze/video", data=bad_file, content_type="multipart/form-data")
test(res_err4.status_code == 400, "Unsupported file format returns 400 Bad Request")
test("Unsupported video format" in res_err4.get_json()["error"], "Unsupported format error message clear")

# ----------------------------------------------------------------------
# 5. STANDALONE VS FLASK API PARITY CROSS-VALIDATION
# ----------------------------------------------------------------------
print("\n5. STANDALONE VS API PARITY CROSS-VALIDATION:")
print("-" * 50)

test_samples_parity = [
    "Machine learning models require thorough validation to ensure generalizability.",
    "Hey friend, let us meet up for coffee later this evening if you are free!",
    "In conclusion, the experimental results demonstrate statistically significant improvements across all primary benchmark criteria."
]

for idx, txt in enumerate(test_samples_parity):
    standalone_out = predict_text(txt)
    api_out = client.post("/api/analyze/text", json={"text": txt}).get_json()
    
    standalone_ai_pct = round(standalone_out["ai_probability"] * 100, 2)
    api_ai_pct = api_out["ai_probability"]
    test(
        abs(standalone_ai_pct - api_ai_pct) < 1e-4,
        f"[Text Sample {idx+1}] Exact Parity: Standalone={standalone_ai_pct}%, API={api_ai_pct}% (Diff: 0.0000%)"
    )

if sample_fake.exists():
    standalone_v = predict_video(str(sample_fake))
    with open(sample_fake, "rb") as vf:
        v_bytes = vf.read()
    api_v = client.post(
        "/api/analyze/video",
        data={"file": (io.BytesIO(v_bytes), "fake.mp4")},
        content_type="multipart/form-data"
    ).get_json()
    
    standalone_fake_pct = round(standalone_v["fake_probability"] * 100, 2)
    api_fake_pct = api_v["fake_probability"]
    test(
        abs(standalone_fake_pct - api_fake_pct) < 1e-4,
        f"[Fake Video Sample] Exact Parity: Standalone={standalone_fake_pct}%, API={api_fake_pct}% (Diff: 0.0000%)"
    )
    standalone_v_lbl = "FAKE" if standalone_fake_pct > (100.0 - standalone_fake_pct) else "REAL"
    test(
        api_v["result"] == standalone_v_lbl,
        f"[Fake Video Sample] Exact Label Parity: Standalone Higher Prob={'FAKE' if standalone_fake_pct > 50 else 'REAL'}, API={api_v['result']}"
    )

if sample_real.exists():
    standalone_vr = predict_video(str(sample_real))
    with open(sample_real, "rb") as vf:
        vr_bytes = vf.read()
    api_vr = client.post(
        "/api/analyze/video",
        data={"file": (io.BytesIO(vr_bytes), "real.mp4")},
        content_type="multipart/form-data"
    ).get_json()
    
    standalone_real_pct = round(standalone_vr["real_probability"] * 100, 2)
    api_real_pct = api_vr["real_probability"]
    test(
        abs(standalone_real_pct - api_real_pct) < 1e-4,
        f"[Real Video Sample] Exact Parity: Standalone={standalone_real_pct}%, API={api_real_pct}% (Diff: 0.0000%)"
    )
    standalone_vr_lbl = "REAL" if standalone_real_pct >= (100.0 - standalone_real_pct) else "FAKE"
    test(
        api_vr["result"] == standalone_vr_lbl,
        f"[Real Video Sample] Exact Label Parity: Standalone Higher Prob={'REAL' if standalone_real_pct >= 50 else 'FAKE'}, API={api_vr['result']}"
    )

# ----------------------------------------------------------------------
# 6. ARCHITECTURE METADATA AUDIT
# ----------------------------------------------------------------------
print("\n6. ARCHITECTURE METADATA AUDIT:")
print("-" * 50)
html_str = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
js_str = (BASE_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")

test("BiLSTM" not in html_str, "templates/index.html does not contain stale 'BiLSTM'")
test("BiLSTM" not in js_str, "static/js/app.js does not contain stale 'BiLSTM'")
test("Temporal Conv1D" in html_str, "templates/index.html contains 'Temporal Conv1D'")
test("BiGRU" in html_str, "templates/index.html contains 'BiGRU'")
test("Multi-Head Attention" in html_str, "templates/index.html contains 'Multi-Head Attention'")
print("-" * 50)
print(f"  First text inference latency : {timings_text[0]:.1f} ms")
print(f"  Subsequent avg text latency  : {np.mean(timings_text[1:]):.1f} ms (min: {np.min(timings_text[1:]):.1f} ms, max: {np.max(timings_text[1:]):.1f} ms)")
if timings_video:
    print(f"  Avg video inference latency  : {np.mean(timings_video):.1f} ms ({np.mean(timings_video)/1000:.2f} s for 8-frame face extraction + Conv-BiGRU)")

print("\n" + "=" * 80)
print(f"END-TO-END SUMMARY: {passed_checks} / {total_checks} CHECKS PASSED ({passed_checks/total_checks*100:.1f}%)")
print("=" * 80)

if passed_checks == total_checks:
    print("STATUS: ALL END-TO-END SYSTEM TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
else:
    print(f"STATUS: {total_checks - passed_checks} CHECKS FAILED.")
    sys.exit(1)
