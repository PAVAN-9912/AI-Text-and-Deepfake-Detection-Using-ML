import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "datasets", "video")
INPUT_CSV = os.path.join(DATA_DIR, "face_temporal_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "video_final")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 2 & 3: DATASET SPLIT VERIFICATION & VALIDATION ERROR ANALYSIS")
print("=" * 80)

# 1. LOAD DATASET & VERIFY SPLITS
print("\n1. LOADING & VERIFYING DATASET SPLITS...")
df = pd.read_csv(INPUT_CSV)
print(f"Total video records: {len(df)}")

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

train_ids = set(train_df["video_id"])
val_ids = set(val_df["video_id"])
test_ids = set(test_df["video_id"])

train_val_overlap = len(train_ids.intersection(val_ids))
train_test_overlap = len(train_ids.intersection(test_ids))
val_test_overlap = len(val_ids.intersection(test_ids))

print(f"  Train samples      : {len(train_df)} (Real: {(train_df['label']==1).sum()}, Fake: {(train_df['label']==0).sum()})")
print(f"  Validation samples : {len(val_df)} (Real: {(val_df['label']==1).sum()}, Fake: {(val_df['label']==0).sum()})")
print(f"  Test samples       : {len(test_df)} (Real: {(test_df['label']==1).sum()}, Fake: {(test_df['label']==0).sum()})")

print(f"\n  Overlap Audit:")
print(f"  - Train & Validation Overlap : {train_val_overlap}")
print(f"  - Train & Test Overlap       : {train_test_overlap}")
print(f"  - Validation & Test Overlap  : {val_test_overlap}")

assert train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0, "Split leakage detected!"
print(">>> SPLIT AUDIT PASSED: ZERO DATA LEAKAGE CONFIRMED ACROSS ALL PARTITIONS. <<<")

# 2. LOAD PRODUCTION MODEL & ARTIFACTS
print("\n2. LOADING PRODUCTION MODEL ARTIFACTS...")
with open(os.path.join(MODEL_DIR, "video_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
feat_mean = scaler["mean"]
feat_std = scaler["std"]

with open(os.path.join(MODEL_DIR, "probability_calibrator.pkl"), "rb") as f:
    calibrator_data = pickle.load(f)
    calibrator = calibrator_data["model"] if isinstance(calibrator_data, dict) else calibrator_data

with open(os.path.join(MODEL_DIR, "video_config.pkl"), "rb") as f:
    config = pickle.load(f)
threshold = config.get("decision_threshold", 0.15)

# Model architecture matching production
class FaceSpatialTemporalModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.25)
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.20
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        proj = self.projection(x)
        gru_out, _ = self.gru(proj)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        mean_pool = norm_out.mean(dim=1)
        max_pool = norm_out.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.classifier(pooled).squeeze(1)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceSpatialTemporalModel(input_dim=512, hidden_dim=128).to(device)
model_path = os.path.join(MODEL_DIR, "temporal_attention_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3. RUN VALIDATION SPLIT INFERENCE
print("\n3. RUNNING VALIDATION SET ERROR INFERENCE (N=1,201)...")
metadata_cols = {
    "split", "label", "label_name", "video_id", "filename",
    "frame_count", "face_detected_frames", "frame_directory"
}
feat_cols = [c for c in df.columns if c not in metadata_cols]

X_val_raw = val_df[feat_cols].values.astype(np.float32).reshape(-1, 8, 512)
y_val = val_df["label"].values.astype(int)

X_val = (X_val_raw - feat_mean) / feat_std

with torch.no_grad():
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_logits = model(val_tensor).cpu().numpy().reshape(-1, 1)

# Apply Platt Calibrator
val_probs = calibrator.predict_proba(val_logits)
p_real = val_probs[:, 1]
p_fake = val_probs[:, 0]

# Predictions with threshold t=0.15 on P(Real)
val_preds = (p_real >= threshold).astype(int)

val_results_df = val_df[["video_id", "filename", "label", "label_name", "face_detected_frames"]].copy()
val_results_df["real_prob"] = np.round(p_real, 4)
val_results_df["fake_prob"] = np.round(p_fake, 4)
val_results_df["confidence"] = np.round(np.where(val_preds == 1, p_real, p_fake), 4)
val_results_df["prediction"] = val_preds
val_results_df["predicted_label"] = np.where(val_preds == 1, "Real", "Fake")
val_results_df["is_correct"] = (val_results_df["label"] == val_results_df["prediction"])

# Error categories:
# False Positive: Fake (0) predicted as Real (1)
# False Negative: Real (1) predicted as Fake (0)
val_results_df["error_category"] = "Correct"
val_results_df.loc[(val_results_df["label"] == 0) & (val_results_df["prediction"] == 1), "error_category"] = "False Positive (Fake -> Real)"
val_results_df.loc[(val_results_df["label"] == 1) & (val_results_df["prediction"] == 0), "error_category"] = "False Negative (Real -> Fake)"

# Compute Validation Metrics
acc = accuracy_score(y_val, val_preds)
bal_acc = balanced_accuracy_score(y_val, val_preds)
prec_real = precision_score(y_val, val_preds, pos_label=1, zero_division=0)
rec_real = recall_score(y_val, val_preds, pos_label=1, zero_division=0)
f1_real = f1_score(y_val, val_preds, pos_label=1, zero_division=0)

prec_fake = precision_score(y_val, val_preds, pos_label=0, zero_division=0)
rec_fake = recall_score(y_val, val_preds, pos_label=0, zero_division=0)
f1_fake = f1_score(y_val, val_preds, pos_label=0, zero_division=0)

auc = roc_auc_score(y_val, p_real)
brier = brier_score_loss(y_val, p_real)

cm = confusion_matrix(y_val, val_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\nValidation Split Performance:")
print(f"  Accuracy          : {acc*100:.2f}% ({tn+tp}/{len(y_val)})")
print(f"  Balanced Accuracy : {bal_acc*100:.2f}%")
print(f"  ROC-AUC           : {auc*100:.2f}%")
print(f"  Real F1           : {f1_real*100:.2f}% (Prec: {prec_real*100:.2f}%, Rec: {rec_real*100:.2f}%)")
print(f"  Fake F1           : {f1_fake*100:.2f}% (Prec: {prec_fake*100:.2f}%, Rec: {rec_fake*100:.2f}%)")
print(f"  Brier Score       : {brier:.5f}")
print(f"  Confusion Matrix  : TN(Fake)={tn}, FP={fp}, FN={fn}, TP(Real)={tp}")

# Error Breakdown
errors_df = val_results_df[~val_results_df["is_correct"]].copy()
fp_df = val_results_df[val_results_df["error_category"] == "False Positive (Fake -> Real)"]
fn_df = val_results_df[val_results_df["error_category"] == "False Negative (Real -> Fake)"]

print(f"\nError Analysis Breakdown:")
print(f"  Total Errors      : {len(errors_df)} / {len(val_df)} (Error Rate: {len(errors_df)/len(val_df)*100:.2f}%)")
print(f"  False Positives   : {len(fp_df)} (Fake predicted as Real)")
print(f"  False Negatives   : {len(fn_df)} (Real predicted as Fake)")

# Analysis by face detection reliability
face_counts = val_results_df.groupby("face_detected_frames")["is_correct"].agg(["count", "mean"]).reset_index()
face_counts["accuracy"] = np.round(face_counts["mean"] * 100, 2)
print(f"\nAccuracy by Face Detected Frames (out of 8):")
print(face_counts[["face_detected_frames", "count", "accuracy"]].to_string(index=False))

# Save Reports
csv_path = os.path.join(REPORTS_DIR, "video_error_analysis.csv")
errors_df.to_csv(csv_path, index=False)
print(f"\nSaved Error Analysis CSV to: {csv_path}")

error_report_json = {
    "validation_metrics": {
        "samples": len(val_df),
        "accuracy": float(round(acc, 4)),
        "balanced_accuracy": float(round(bal_acc, 4)),
        "roc_auc": float(round(auc, 4)),
        "real_metrics": {
            "precision": float(round(prec_real, 4)),
            "recall": float(round(rec_real, 4)),
            "f1": float(round(f1_real, 4))
        },
        "fake_metrics": {
            "precision": float(round(prec_fake, 4)),
            "recall": float(round(rec_fake, 4)),
            "f1": float(round(f1_fake, 4))
        },
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "brier_score": float(round(brier, 5)),
        "threshold": float(threshold)
    },
    "errors_summary": {
        "total_errors": len(errors_df),
        "false_positives_fake_as_real": len(fp_df),
        "false_negatives_real_as_fake": len(fn_df),
        "by_face_detected_frames": face_counts.to_dict(orient="records")
    }
}

json_path = os.path.join(REPORTS_DIR, "video_error_analysis.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(error_report_json, f, indent=2)
print(f"Saved Error Analysis JSON to: {json_path}")

print("=" * 80)
print("PHASE 2 & 3 COMPLETED SUCCESSFULLY!")
print("=" * 80)
