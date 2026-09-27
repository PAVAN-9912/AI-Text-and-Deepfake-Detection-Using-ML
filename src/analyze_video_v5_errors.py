import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "datasets", "video")
INPUT_CSV = os.path.join(DATA_DIR, "face_temporal_features.csv")
V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "video_final_v4")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("STEP 1: DETAILED V4 VALIDATION ERROR ANALYSIS")
print("=" * 80)

# Load feature dataset
df = pd.read_csv(INPUT_CSV)
val_df = df[df["split"] == "validation"].copy()

metadata_cols = {
    "split", "label", "label_name", "video_id", "filename",
    "frame_count", "face_detected_frames", "frame_directory"
}
feat_cols = [c for c in df.columns if c not in metadata_cols]

X_val_raw = val_df[feat_cols].values.astype(np.float32).reshape(-1, 8, 512)
y_val = val_df["label"].values.astype(int)

# Load V4 Scaler & Calibrator & Model
with open(os.path.join(V4_MODEL_DIR, "video_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
feat_mean = scaler["mean"]
feat_std = scaler["std"]

with open(os.path.join(V4_MODEL_DIR, "probability_calibrator.pkl"), "rb") as f:
    cal_data = pickle.load(f)
    calibrator = cal_data["model"] if isinstance(cal_data, dict) else cal_data
    threshold = float(cal_data.get("threshold", 0.32))

from src.predict_video_final_v4 import TemporalConvBiGRUModel, DEVICE

model = TemporalConvBiGRUModel(input_dim=512, hidden_dim=128).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(V4_MODEL_DIR, "temporal_attention_model.pth"), map_location=DEVICE))
model.eval()

# Standardize features
X_val = (X_val_raw - feat_mean) / feat_std

with torch.no_grad():
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    logits = model(val_tensor).cpu().numpy().reshape(-1, 1)

probs = calibrator.predict_proba(logits)
p_fake = probs[:, 0]
p_real = probs[:, 1]
preds = (p_real >= threshold).astype(int)

val_df["p_fake"] = np.round(p_fake, 4)
val_df["p_real"] = np.round(p_real, 4)
val_df["predicted_label"] = np.where(preds == 1, 1, 0)
val_df["predicted_class"] = np.where(preds == 1, "Real", "Fake")
val_df["is_correct"] = (val_df["label"] == val_df["predicted_label"])
val_df["margin"] = np.round(np.abs(p_real - threshold), 4)

# Error categorization
# Real -> Fake (Label 1 predicted as 0): False Negative on Real
# Fake -> Real (Label 0 predicted as 1): False Positive on Real
val_df["error_type"] = "Correct"
val_df.loc[(val_df["label"] == 1) & (val_df["predicted_label"] == 0), "error_type"] = "Real_As_Fake_FN"
val_df.loc[(val_df["label"] == 0) & (val_df["predicted_label"] == 1), "error_type"] = "Fake_As_Real_FP"

acc = accuracy_score(y_val, preds)
bal_acc = balanced_accuracy_score(y_val, preds)
auc = roc_auc_score(y_val, p_real)
f1_real = f1_score(y_val, preds, pos_label=1, zero_division=0)
f1_fake = f1_score(y_val, preds, pos_label=0, zero_division=0)
rec_real = recall_score(y_val, preds, pos_label=1, zero_division=0)
rec_fake = recall_score(y_val, preds, pos_label=0, zero_division=0)

cm = confusion_matrix(y_val, preds)
tn, fp, fn, tp = cm.ravel()

print(f"Validation Performance for V4:")
print(f"  Accuracy          : {acc*100:.2f}% ({tn+tp}/{len(y_val)})")
print(f"  Balanced Accuracy : {bal_acc*100:.2f}%")
print(f"  ROC-AUC           : {auc*100:.2f}%")
print(f"  Real Recall       : {rec_real*100:.2f}% | Real F1: {f1_real*100:.2f}%")
print(f"  Fake Recall       : {rec_fake*100:.2f}% | Fake F1: {f1_fake*100:.2f}%")
print(f"  Confusion Matrix  : TN(Fake)={tn}, FP={fp}, FN={fn}, TP(Real)={tp}")
print(f"  Total Errors      : {fp+fn} (Real->Fake: {fn}, Fake->Real: {fp})")

# Save Error CSV
cols_to_save = [
    "video_id", "filename", "label", "label_name", "predicted_label", "predicted_class",
    "p_fake", "p_real", "margin", "error_type", "face_detected_frames", "frame_count"
]
errors_df = val_df[~val_df["is_correct"]][cols_to_save].copy()
csv_path = os.path.join(REPORTS_DIR, "video_v5_error_analysis.csv")
errors_df.to_csv(csv_path, index=False)
print(f"\nSaved V5 Error Analysis CSV: {csv_path}")

# Breakdown by face detection stability
face_counts = val_df.groupby("face_detected_frames")["is_correct"].agg(["count", "mean"]).reset_index()
face_counts["accuracy"] = np.round(face_counts["mean"] * 100, 2)

# High confidence mistakes
high_conf_errors = errors_df[errors_df["margin"] > 0.35].copy()

error_report_json = {
    "v4_validation_summary": {
        "accuracy": float(round(acc, 4)),
        "balanced_accuracy": float(round(bal_acc, 4)),
        "roc_auc": float(round(auc, 4)),
        "real_recall": float(round(rec_real, 4)),
        "real_f1": float(round(f1_real, 4)),
        "fake_recall": float(round(rec_fake, 4)),
        "fake_f1": float(round(f1_fake, 4)),
        "total_errors": int(fp + fn),
        "real_as_fake_errors": int(fn),
        "fake_as_real_errors": int(fp),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "threshold": float(threshold)
    },
    "face_detection_accuracy": face_counts.to_dict(orient="records"),
    "high_confidence_mistakes_count": len(high_conf_errors)
}

json_path = os.path.join(REPORTS_DIR, "video_v5_error_analysis.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(error_report_json, f, indent=2)
print(f"Saved V5 Error Analysis JSON: {json_path}")
print("=" * 80)
