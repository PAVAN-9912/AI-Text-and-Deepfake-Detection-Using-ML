"""
Phase 6: Independent Evaluation of Vision Transformer (ViT) on the Official Test Set
====================================================================================
Evaluates the fine-tuned Vision Transformer (google/vit-base-patch16-224)
on the 518-video official test set using the calibrated threshold from calibrate_vit.py.
"""

import sys
import io
import csv
import json
import logging
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score
)

BASE = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE / "experiments" / "v12_vit"
DATA_DIR = BASE / "experiments" / "v11_multigenerator" / "data"
RES_DIR = EXP_DIR / "results"
REPORTS_DIR = EXP_DIR / "reports"
RES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))

from src.predict_video_vit import predict_video_vit, load_vit_artifacts, _threshold as current_threshold


def main():
    load_vit_artifacts()
    # Read threshold from calibration json
    cal_file = EXP_DIR / "checkpoints" / "vit_threshold.json"
    if cal_file.exists():
        with open(cal_file, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        threshold = float(cal_data.get("threshold", 0.50))
    else:
        threshold = float(current_threshold)

    print(f"ViT inference module ready | Active Calibrated Threshold = {threshold:.4f}")

    manifest_path = DATA_DIR / "v11_final_test_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Final test manifest missing at: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Final test set: {len(rows)} videos")
    n_real = sum(1 for r in rows if r["label_name"] == "real")
    n_fake = sum(1 for r in rows if r["label_name"] == "fake")
    print(f"  Ground Truth Counts: Real = {n_real} | Fake = {n_fake}\n")

    labels = []
    probs = []
    results = []
    n_insuf = 0

    for i, row in enumerate(rows):
        vpath = Path(row["video_path"])
        true_label = int(row["label"])  # 1 = Real, 0 = Fake

        result = predict_video_vit(str(vpath))

        label_out = result["label"]
        real_prob = result["real_probability"]
        reliability = result["reliability"]

        if label_out in ("INSUFFICIENT_EVIDENCE", "ERROR"):
            n_insuf += 1
            pred_label = 0  # Conservative fallback
            real_prob_use = 0.0
        else:
            pred_label = 1 if real_prob >= threshold else 0
            real_prob_use = real_prob

        labels.append(true_label)
        probs.append(real_prob_use)

        correct = (pred_label == true_label)
        results.append({
            "video": vpath.name,
            "source": row["source"],
            "true_label": row["label_name"],
            "pred_label": "real" if pred_label == 1 else "fake",
            "real_prob": round(real_prob_use, 4),
            "correct": int(correct),
            "reliability": reliability,
            "face_count": result.get("face_count", 0),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            acc_so_far = np.mean([r["correct"] for r in results])
            print(f"  [{i+1:3d}/{len(rows)}] Running Accuracy = {acc_so_far:.4f}")

    labels_np = np.array(labels)
    probs_np = np.array(probs)
    preds_np = (probs_np >= threshold).astype(int)

    acc = float(np.mean(preds_np == labels_np))
    bal_acc = balanced_accuracy_score(labels_np, preds_np)
    auc = roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else float("nan")
    cm = confusion_matrix(labels_np, preds_np)
    tn, fp, fn, tp = cm.ravel()
    real_rec = tp / max(tp + fn, 1)
    fake_rec = tn / max(tn + fp, 1)
    prec_real = tp / max(tp + fp, 1)
    prec_fake = tn / max(tn + fn, 1)
    f1_real = 2 * prec_real * real_rec / max(prec_real + real_rec, 1e-6)
    f1_fake = 2 * prec_fake * fake_rec / max(prec_fake + fake_rec, 1e-6)

    print(f"\n{'='*70}")
    print("VISION TRANSFORMER (ViT) FINAL INDEPENDENT TEST EVALUATION")
    print(f"{'='*70}")
    print(f"  Total videos        : {len(rows)}")
    print(f"  Threshold           : {threshold:.4f}")
    print(f"  Overall Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy   : {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  Real Recall (Sens)  : {real_rec:.4f} ({real_rec*100:.2f}%)")
    print(f"  Fake Recall (Spec)  : {fake_rec:.4f} ({fake_rec*100:.2f}%)")
    print(f"  Real Precision      : {prec_real:.4f}")
    print(f"  Fake Precision      : {prec_fake:.4f}")
    print(f"  F1-Score (Real)     : {f1_real:.4f}")
    print(f"  F1-Score (Fake)     : {f1_fake:.4f}")
    print(f"  Confusion Matrix    : TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Zero-face/Insuf     : {n_insuf}")
    print(f"{'='*70}\n")

    # Save predictions
    pred_csv = RES_DIR / "vit_final_test_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved prediction records to: {pred_csv}")

    # Save summary
    summary = {
        "model": "Vision Transformer (google/vit-base-patch16-224)",
        "threshold": threshold,
        "test_set": "Official Celeb-DF v2 (518 videos)",
        "n_videos": len(rows),
        "n_real": n_real,
        "n_fake": n_fake,
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "roc_auc": round(auc, 6) if not np.isnan(auc) else None,
        "real_recall": round(real_rec, 6),
        "fake_recall": round(fake_rec, 6),
        "real_precision": round(prec_real, 6),
        "fake_precision": round(prec_fake, 6),
        "f1_real": round(f1_real, 6),
        "f1_fake": round(f1_fake, 6),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "n_insufficient_evidence": n_insuf,
    }
    summary_path = RES_DIR / "vit_final_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
