# AI Content Detection Platform — Live Demonstration Guide

This guide provides verified step-by-step instructions for launching and demonstrating the **AI Content Detection Platform** (`PolicyC_STSeqNet_FaceViT` + `TextDetector V3.4`) during live evaluations and viva examinations.

---

## 1. Quick Start & Launch Instructions

### Prerequisites
- **OS:** Windows 10/11 (or Linux/macOS)
- **Python:** Python 3.11.9 virtual environment (`venv`)
- **Hardware:** CPU (`torch 2.13.0+cpu`, ~1.0–1.3 GB RAM during inference) or CUDA GPU

### Launch Commands
```powershell
cd "C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection"
.\venv\Scripts\python.exe app.py
```

*Expected Startup Output:*
```text
Text Model    : TextDetector V3.4 (models/text_final/)
Video Model   : VideoDetector ViT Base (experiments/v12_vit/) [Active]
 * Running on http://127.0.0.1:5000
```
*(Active production video routing uses `PolicyC_STSeqNet_FaceViT` as reported by `GET /api/health`.)*

Open your browser to: `http://127.0.0.1:5000`

---

## 2. Text Detection Demonstration Walkthrough

Navigate to the **Text Detection** tab (`#text`).

### Demo Case T1: Human-Written Technical Narrative
**Input (101 words):**
```text
When we first set up the server rack in the basement of the engineering building, nobody expected the cooling fans to sound quite like a small turboprop aircraft. We spent three afternoons routing Cat6 cables behind the steel conduit, labeling every patch panel port with masking tape because the thermal labeler had jammed twice. By Friday evening the switches were blinking green, the UPS batteries were topped off, and the lab could finally run overnight builds without tripping the hallway breaker. Looking back, half the trouble came from a single misconfigured VLAN trunk that dropped packets only under heavy NFS load.
```
**Observed Result:** `HUMAN-WRITTEN` (`Confidence: 81.72%`, `AI Probability: 18.28%`, `length_group: long`, `threshold: 0.59`).

### Demo Case T2: AI-Generated Corporate Text
**Input (46 words):**
```text
In today's fast-paced digital landscape, artificial intelligence has revolutionized numerous industries by streamlining workflows and augmenting human creativity. Furthermore, leveraging cutting-edge machine learning algorithms enables organizations to optimize operational efficiency, enhance decision-making frameworks, and foster unprecedented innovation. In conclusion, embracing these transformative technological advancements is imperative.
```
**Observed Result:** `AI-GENERATED` (`Confidence: 99.08%`, `AI Probability: 99.08%`, `length_group: medium`, `threshold: 0.64`).

### Demo Case T3: Edge Case — Very Short Text (`< 3` Words)
**Input:** `Hello there!`  
**Observed Result:** `INSUFFICIENT_EVIDENCE` (Unified verdict `INCONCLUSIVE`, `word_count: 2`, `Confidence: N/A`).

---

## 3. Video Detection Demonstration Walkthrough

Navigate to the **Video Detection** tab (`#video`).

### Demo Case V1: Deepfake Face-Swap (`FAKE`)
- **File:** `datasets/video/celebdf_raw/Celeb-synthesis/id0_id16_0003.mp4`
- **Observed Result:** `FAKE` (`C2_DECISIVE_FACIAL_MANIPULATION`)
- **Expert Scores:** `ST-SeqNet = 0.9941`, `FaceViT = 0.6811` (`8` faces detected), latency `~1.0–1.6s`.

### Demo Case V2: Faceless AI-Generated Video (`FAKE`)
- **File:** `datasets/video/vidprom_subset/pika/pika-19b1aaa0-cd33-5195-96da-3b138bcef711.mp4`
- **Observed Result:** `FAKE` (`C1_FACELESS_AI_AUTHORITY`)
- **Expert Scores:** `ST-SeqNet = 0.8872`, `FaceViT = None` (`0` faces detected), latency `~0.76s`.

### Demo Case V3: Genuine Faceless Camera Video (`REAL`)
- **File:** `datasets/video/external_phase7/primary/EXT_REAL_CAM_NOFACE_001.webm`
- **Observed Result:** `REAL` (`C1_FACELESS_GENUINE_AUTHORITY`)
- **Expert Scores:** `ST-SeqNet = 0.0003`, `FaceViT = None` (`0` faces detected), latency `~1.25s`.

### Demo Case V4: Policy C Domain Conflict Abstention (`INCONCLUSIVE`)
- **File:** `datasets/video/celebdf_raw/YouTube-real/00000.mp4`
- **Observed Result:** `INCONCLUSIVE` (`C3_STRONG_DOMAIN_CONFLICT`)
- **Expert Scores:** `ST-SeqNet = 0.9968` vs. `FaceViT = 0.0047` (`8` faces detected), latency `~0.93–1.01s`.
- **Significance:** Demonstrates calibrated safety—when experts strongly conflict (`FaceViT <= 0.20` vs `ST-SeqNet >= 0.80`), Policy C abstains instead of producing a false positive.

---

## 4. Security & Robustness Checks
- **Empty / Corrupt / Unsupported Upload:** Uploading `.txt`, `0`-byte `.mp4`, or random bytes returns `HTTP 400` with sanitized JSON errors and deletes temporary files in `finally:`.
- **Oversized Upload (`> 50 MB`):** Rejected with `HTTP 413`.
