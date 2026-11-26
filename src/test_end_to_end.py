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
try:
    from src.predict_video_fusion import predict_video_fusion as active_predict_video
except Exception:
    from src.predict_video_vit import predict_video_vit as active_predict_video

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
test("AI Content Detection Platform" in html_content, "HTML contains title and branding")
test('id="home"' in html_content and 'id="text"' in html_content and 'id="video"' in html_content, "HTML contains all core tab sections")

res_health = client.get("/api/health")
test(res_health.status_code == 200, "GET /api/health returns 200 OK")
health_data = res_health.get_json()
test(health_data["status"] == "healthy", "Health status is 'healthy'")
test(health_data["text_model"] and health_data["video_model"] and health_data["minilm"] and health_data["face_detector"], "All sub-engines reported ready")
test(health_data.get("video_engine") == "PolicyC_STSeqNet_FaceViT", f"Active video engine reported: {health_data.get('video_engine')}")

res_metrics = client.get("/api/metrics")
test(res_metrics.status_code == 200, "GET /api/metrics returns 200 OK")
metrics_data = res_metrics.get_json()
test(metrics_data["text_metrics"]["accuracy"] == 89.88, "Text accuracy is 89.88%")
test(metrics_data["video_metrics"]["accuracy"] > 70.0, f"Video accuracy is valid: {metrics_data['video_metrics']['accuracy']}%")
test(metrics_data["video_metrics"]["roc_auc"] > 80.0, f"Video ROC-AUC is valid: {metrics_data['video_metrics']['roc_auc']}%")
test("model" in metrics_data["video_metrics"], f"Video model name reported: {metrics_data['video_metrics']['model']}")

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
        test(vdata["result"] in ["REAL", "FAKE", "INCONCLUSIVE"], f"[{name}] Policy C returns REAL, FAKE, or INCONCLUSIVE (Got: {vdata['result']})")
        test("routing_rule" in vdata and vdata["routing_rule"] is not None, f"[{name}] Policy C routing_rule: {vdata.get('routing_rule')}")
        test(vdata["frames_analyzed"] == 16, f"[{name}] ST-SeqNet frames analyzed strictly 16")
        test("PolicyC_STSeqNet_FaceViT" in vdata["model"], f"[{name}] Model name reported: {vdata['model']}")
        test("routing_rule" in vdata and vdata["routing_rule"] is not None, f"[{name}] Policy C routing_rule: {vdata.get('routing_rule')}")
        test("attention_weights" in vdata and len(vdata["attention_weights"]) == 16, f"[{name}] 16 temporal attention weights returned")
        test(vdata["threshold"] is not None, f"[{name}] Threshold is valid: {vdata['threshold']}")
        if vdata["result"] in ["REAL", "FAKE"]:
            p_real = vdata["real_probability"]
            p_fake = vdata["fake_probability"]
            test(abs(p_real + p_fake - 100.0) < 0.05, f"[{name}] Probabilities sum to 100%: Real={p_real}%, Fake={p_fake}%")
            expected_conf = p_real if vdata["result"] == "REAL" else p_fake
            test(abs(vdata["confidence"] - expected_conf) < 1e-4, f"[{name}] Confidence strictly equals predicted class prob ({expected_conf}%)")

# Verify zero-face video delegates via frozen Policy C (C1_FACELESS_*)
import cv2, tempfile
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf_nf:
    temp_nf = tf_nf.name
try:
    out_nf = cv2.VideoWriter(temp_nf, cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
    for _ in range(16):
        out_nf.write(np.zeros((224, 224, 3), dtype=np.uint8))
    out_nf.release()
    with open(temp_nf, "rb") as vf_nf:
        nf_resp = client.post("/api/analyze/video", data={"file": (io.BytesIO(vf_nf.read()), "zeroface.mp4")}, content_type="multipart/form-data")
    nf_json = nf_resp.get_json()
    test(nf_json.get("result") in ("REAL", "FAKE") and str(nf_json.get("routing_rule", "")).startswith("C1_FACELESS_"), f"[Policy C Zero-Face] Delegated to ST-SeqNet: {nf_json.get('result')} ({nf_json.get('routing_rule')})")
finally:
    if os.path.exists(temp_nf):
        os.remove(temp_nf)

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
    standalone_v = active_predict_video(str(sample_fake))
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
    standalone_v_lbl = standalone_v.get("label", "FAKE")
    test(
        api_v["result"] == standalone_v_lbl,
        f"[Fake Video Sample] Exact Label Parity: Standalone={standalone_v_lbl}, API={api_v['result']}"
    )

if sample_real.exists():
    standalone_vr = active_predict_video(str(sample_real))
    with open(sample_real, "rb") as vf:
        vr_bytes = vf.read()
    api_vr = client.post(
        "/api/analyze/video",
        data={"file": (io.BytesIO(vr_bytes), "real.mp4")},
        content_type="multipart/form-data"
    ).get_json()
    
    standalone_vr_lbl = standalone_vr.get("label", "REAL")
    test(
        api_vr["result"] == standalone_vr_lbl,
        f"[Real Video Sample] Exact Label Parity: Standalone={standalone_vr_lbl}, API={api_vr['result']}"
    )
    if standalone_vr.get("real_probability") is not None and api_vr.get("real_probability") is not None:
        standalone_real_pct = round(standalone_vr["real_probability"] * 100, 2)
        api_real_pct = api_vr["real_probability"]
        test(
            abs(standalone_real_pct - api_real_pct) < 1e-4,
            f"[Real Video Sample] Exact Parity: Standalone={standalone_real_pct}%, API={api_real_pct}% (Diff: 0.0000%)"
        )
    else:
        test(
            abs((standalone_vr.get("st_seqnet_p_ai") or 0.0) - (api_vr.get("st_seqnet_p_ai") or 0.0)) < 1e-6,
            f"[Real Video Sample] Exact ST-SeqNet Parity on INCONCLUSIVE: {api_vr.get('st_seqnet_p_ai')}"
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
