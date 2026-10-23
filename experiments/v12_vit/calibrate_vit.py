"""
ViT Threshold Calibration Module
================================
Calibrates the operating threshold for Vision Transformer (ViT) deepfake detector
using the disjoint calibration dataset.

Threshold optimization objective:
  Maximize Balanced Accuracy = (Real Recall + Fake Recall) / 2
"""

import sys
import os
import io
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import ViTForImageClassification
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "experiments" / "v12_vit"
DATA_DIR = BASE_DIR / "experiments" / "v11_multigenerator" / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
RES_DIR = EXP_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

CROPS_DIR = DATA_DIR / "crops" / "calibration"
MANIFEST = DATA_DIR / "v11_calibration_manifest.csv"
CKPT_PATH = CKPT_DIR / "vit_deepfake_best.pth"
OUT_THRESHOLD_JSON = CKPT_DIR / "vit_threshold.json"

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]
INPUT_SIZE = 224
DEVICE = torch.device("cpu")

transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


def load_model(ckpt_path):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
        ignore_mismatched_sizes=True,
    )
    state = torch.load(str(ckpt_path), map_location=DEVICE)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, state


def load_calibration_crops(crops_dir, manifest_path):
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    video_label = {}
    for r in rows:
        stem = Path(r["video_path"]).stem
        video_label[stem] = int(r["label"])

    groups = {}
    for label_name in ["real", "fake"]:
        ldir = crops_dir / label_name
        if not ldir.exists():
            continue
        for cp in sorted(ldir.iterdir()):
            if cp.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = cp.stem
            for suffix in ["_crop", "_nf"]:
                idx = stem.rfind(suffix)
                if idx > 0:
                    stem = stem[:idx]
                    break
            if stem not in groups:
                groups[stem] = {"crops": [], "label": video_label.get(stem, -1)}
            if "_nf" not in cp.stem:
                groups[stem]["crops"].append(cp)

    valid = [(s, g["label"], g["crops"])
             for s, g in groups.items()
             if g["label"] in (0, 1) and len(g["crops"]) > 0]
    print(f"Calibration videos with face crops: {len(valid)}")
    return valid


def calibrate():
    print("=" * 70)
    print("CALIBRATING ViT DECISION THRESHOLD ON CALIBRATION DATASET")
    print("=" * 70)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")

    model, state = load_model(CKPT_PATH)
    video_groups = load_calibration_crops(CROPS_DIR, MANIFEST)

    true_labels = []
    video_real_probs = []

    print(f"Running inference on {len(video_groups)} calibration videos...")
    with torch.no_grad():
        for i, (stem, label, crop_paths) in enumerate(video_groups):
            imgs = [transform(Image.open(p).convert("RGB")) for p in crop_paths]
            batch = torch.stack(imgs).to(DEVICE)
            outputs = model(batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            p_real_crops = probs[:, 1].cpu().numpy()
            mean_p_real = float(np.mean(p_real_crops))

            true_labels.append(label)
            video_real_probs.append(mean_p_real)

            if (i + 1) % 50 == 0 or (i + 1) == len(video_groups):
                print(f"  Processed {i+1}/{len(video_groups)} videos")

    y_true = np.array(true_labels)
    y_prob = np.array(video_real_probs)

    auc = roc_auc_score(y_true, y_prob)
    print(f"\nCalibration Video-Level ROC-AUC: {auc:.4f}")

    # Threshold sweep
    best_thr = 0.50
    best_bal_acc = 0.0
    best_stats = {}

    thresholds = np.linspace(0.1, 0.9, 161)
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        real_recall = tp / max(tp + fn, 1)
        fake_recall = tn / max(tn + fp, 1)
        bal_acc = (real_recall + fake_recall) / 2.0
        acc = (tp + tn) / len(y_true)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = float(thr)
            best_stats = {
                "threshold": round(best_thr, 4),
                "balanced_accuracy": round(float(bal_acc), 4),
                "accuracy": round(float(acc), 4),
                "real_recall": round(float(real_recall), 4),
                "fake_recall": round(float(fake_recall), 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "auc": round(float(auc), 4),
            }

    print("\n" + "=" * 50)
    print("OPTIMAL CALIBRATED THRESHOLD FOUND:")
    print(f"  Threshold          : {best_stats['threshold']:.4f}")
    print(f"  Balanced Accuracy  : {best_stats['balanced_accuracy']:.4f}")
    print(f"  Overall Accuracy   : {best_stats['accuracy']:.4f}")
    print(f"  Real Recall (Sens) : {best_stats['real_recall']:.4f} (TP={best_stats['tp']}, FN={best_stats['fn']})")
    print(f"  Fake Recall (Spec) : {best_stats['fake_recall']:.4f} (TN={best_stats['tn']}, FP={best_stats['fp']})")
    print(f"  ROC-AUC            : {best_stats['auc']:.4f}")
    print("=" * 50)

    with open(OUT_THRESHOLD_JSON, "w", encoding="utf-8") as f:
        json.dump(best_stats, f, indent=2)
    print(f"Saved threshold metadata to: {OUT_THRESHOLD_JSON}")


if __name__ == "__main__":
    calibrate()
