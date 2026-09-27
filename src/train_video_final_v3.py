import os
import sys
import copy
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = r"C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection"
DATA_DIR = os.path.join(BASE_DIR, "datasets", "video")
INPUT_CSV = os.path.join(DATA_DIR, "face_temporal_features.csv")
MODEL_FINAL_DIR = os.path.join(BASE_DIR, "models", "video_final")
os.makedirs(MODEL_FINAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("PHASE 2: TRAINING FACE-AWARE TEMPORAL DEEPFAKE DETECTOR (V3)")
print("=============================================================")
print(f"Device: {DEVICE}")

# 1. LOAD DATA
print("\n1. LOADING FACE-CROPPED TEMPORAL FEATURES...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} video records.")

metadata_cols = {
    "split", "label", "label_name", "video_id", "filename",
    "frame_count", "face_detected_frames", "frame_directory"
}
feat_cols = [c for c in df.columns if c not in metadata_cols]
assert len(feat_cols) == 8 * 512, f"Expected 4096 feature columns, got {len(feat_cols)}"

# Split masks
train_mask = (df["split"] == "train").values
val_mask = (df["split"] == "validation").values
test_mask = (df["split"] == "test").values

X_raw = df[feat_cols].values.astype(np.float32).reshape(-1, 8, 512)
y_all = df["label"].values.astype(np.float32)

X_train_raw = X_raw[train_mask]
y_train = y_all[train_mask]

X_val_raw = X_raw[val_mask]
y_val = y_all[val_mask]

X_test_raw = X_raw[test_mask]
y_test = y_all[test_mask]

print(f"Train split     : {X_train_raw.shape} | Fake(0): {(y_train==0).sum()}, Real(1): {(y_train==1).sum()}")
print(f"Validation split: {X_val_raw.shape} | Fake(0): {(y_val==0).sum()}, Real(1): {(y_val==1).sum()}")
print(f"Test split      : {X_test_raw.shape} | Fake(0): {(y_test==0).sum()}, Real(1): {(y_test==1).sum()}")

# 2. FEATURE STANDARDIZATION (Fitted on Train Only)
print("\n2. STANDARDIZING FEATURES (TRAINING ONLY)...")
train_flat = X_train_raw.reshape(-1, 512)
feat_mean = train_flat.mean(axis=0)
feat_std = train_flat.std(axis=0)
feat_std[feat_std < 1e-6] = 1.0

X_train = (X_train_raw - feat_mean) / feat_std
X_val = (X_val_raw - feat_mean) / feat_std
X_test = (X_test_raw - feat_mean) / feat_std

# Save Scaler
scaler_artifact = {"mean": feat_mean, "std": feat_std}
with open(os.path.join(MODEL_FINAL_DIR, "video_scaler.pkl"), "wb") as f:
    pickle.dump(scaler_artifact, f)

# 3. DATA LOADERS
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. ENHANCED TEMPORAL ARCHITECTURE
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
        
        # Classifier with dual pooling (mean + max)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        proj = self.projection(x) # (B, 8, 256)
        gru_out, _ = self.gru(proj) # (B, 8, 256)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out) # (B, 8, 256)
        norm_out = self.norm(gru_out + attn_out) # (B, 8, 256)
        
        # Temporal dual pooling
        mean_pool = norm_out.mean(dim=1) # (B, 256)
        max_pool = norm_out.max(dim=1).values # (B, 256)
        pooled = torch.cat([mean_pool, max_pool], dim=1) # (B, 512)
        
        logits = self.classifier(pooled).squeeze(1) # (B)
        return logits

# Initialize Model, Loss, Optimizer
model = FaceSpatialTemporalModel(input_dim=512, hidden_dim=128).to(DEVICE)

# Calculate class imbalance weight: Neg / Pos = 4240 / 570 = 7.438
pos_weight_val = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
print(f"Class imbalance weight pos_weight = {pos_weight_val:.3f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
epochs = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

# 5. TRAINING LOOP WITH VALIDATION ROC-AUC CHECKPOINTING
print("\n3. TRAINING MODEL...")
best_val_auc = 0.0
best_val_bacc = 0.0
best_model_state = None
best_epoch = 0

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    
    scheduler.step()
    train_loss = total_loss / len(train_dataset)

    # Validation evaluation
    model.eval()
    val_logits_list = []
    with torch.no_grad():
        for X_vb, _ in val_loader:
            X_vb = X_vb.to(DEVICE)
            val_logits_list.append(model(X_vb).cpu())
    
    val_logits = torch.cat(val_logits_list).numpy()
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_auc = roc_auc_score(y_val, val_probs)
    val_preds_05 = (val_probs >= 0.5).astype(int)
    val_bacc = balanced_accuracy_score(y_val, val_preds_05)
    val_f1 = f1_score(y_val, val_preds_05, zero_division=0)

    print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {val_auc*100:.2f}% | Val BalAcc: {val_bacc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_val_bacc = val_bacc
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())

print(f"\nBest Model Checkpoint: Epoch {best_epoch} (Validation ROC-AUC: {best_val_auc*100:.2f}%)")
model.load_state_dict(best_model_state)

# Save Model Weights
torch.save(best_model_state, os.path.join(MODEL_FINAL_DIR, "temporal_attention_model.pth"))

# 6. VALIDATION THRESHOLD CALIBRATION & PLATT SCALING
print("\n4. CALIBRATING THRESHOLD & PLATT SCALING ON VALIDATION SET...")
model.eval()
with torch.no_grad():
    val_logits = []
    for X_vb, _ in val_loader:
        val_logits.append(model(X_vb.to(DEVICE)).cpu())
    val_logits = torch.cat(val_logits).numpy().reshape(-1, 1)

val_probs_raw = 1.0 / (1.0 + np.exp(-val_logits.flatten()))

# Fit Platt Calibrator
platt_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
platt_calibrator.fit(val_logits, y_val.astype(int))

val_probs_calibrated = platt_calibrator.predict_proba(val_logits)[:, 1]
brier_before = brier_score_loss(y_val, val_probs_raw)
brier_after = brier_score_loss(y_val, val_probs_calibrated)
print(f"Validation Brier Score: Raw={brier_before:.5f} -> Platt Calibrated={brier_after:.5f}")

# Threshold sweep on validation
best_thresh = 0.50
best_thresh_bacc = 0.0
best_thresh_f1 = 0.0
threshold_rows = []

for t in np.arange(0.15, 0.86, 0.01):
    t_val = round(float(t), 2)
    preds_t = (val_probs_calibrated >= t_val).astype(int)
    acc_t = accuracy_score(y_val, preds_t)
    bacc_t = balanced_accuracy_score(y_val, preds_t)
    f1_t = f1_score(y_val, preds_t, zero_division=0)
    rec_real = recall_score(y_val, preds_t, pos_label=1, zero_division=0)
    rec_fake = recall_score(y_val, preds_t, pos_label=0, zero_division=0)
    threshold_rows.append({
        "threshold": t_val, "accuracy": acc_t, "balanced_accuracy": bacc_t,
        "f1": f1_t, "real_recall": rec_real, "fake_recall": rec_fake
    })
    # Balance metric: max balanced accuracy with stable F1
    if bacc_t > best_thresh_bacc:
        best_thresh_bacc = bacc_t
        best_thresh_f1 = f1_t
        best_thresh = t_val

print(f"Selected Frozen Threshold from Validation: {best_thresh:.2f} (Val BalAcc: {best_thresh_bacc*100:.2f}%, Val F1: {best_thresh_f1*100:.2f}%)")

# Save Probability Calibrator & Threshold Config
calibrator_data = {"type": "platt_sigmoid", "model": platt_calibrator, "threshold": best_thresh}
with open(os.path.join(MODEL_FINAL_DIR, "probability_calibrator.pkl"), "wb") as f:
    pickle.dump(calibrator_data, f)

# 7. FINAL VALIDATION METRICS (AT FROZEN THRESHOLD)
val_final_preds = (val_probs_calibrated >= best_thresh).astype(int)
val_acc = accuracy_score(y_val, val_final_preds)
val_bacc = balanced_accuracy_score(y_val, val_final_preds)
val_prec_real = precision_score(y_val, val_final_preds, pos_label=1, zero_division=0)
val_rec_real = recall_score(y_val, val_final_preds, pos_label=1, zero_division=0)
val_f1_real = f1_score(y_val, val_final_preds, pos_label=1, zero_division=0)
val_prec_fake = precision_score(y_val, val_final_preds, pos_label=0, zero_division=0)
val_rec_fake = recall_score(y_val, val_final_preds, pos_label=0, zero_division=0)
val_f1_fake = f1_score(y_val, val_final_preds, pos_label=0, zero_division=0)
val_cm = confusion_matrix(y_val, val_final_preds)

print("\n" + "=" * 70)
print("FINAL MODEL — VALIDATION SET METRICS (N=1,201)")
print("=" * 70)
print(f"Accuracy         : {val_acc*100:.2f}%")
print(f"Balanced Accuracy: {val_bacc*100:.2f}%")
print(f"ROC-AUC          : {best_val_auc*100:.2f}%")
print(f"Real Class (1)   : Precision={val_prec_real*100:.2f}%, Recall={val_rec_real*100:.2f}%, F1={val_f1_real*100:.2f}%")
print(f"Fake Class (0)   : Precision={val_prec_fake*100:.2f}%, Recall={val_rec_fake*100:.2f}%, F1={val_f1_fake*100:.2f}%")
print(f"Confusion Matrix : TN(Fake)={val_cm[0,0]}, FP(Fake->Real)={val_cm[0,1]}, FN(Real->Fake)={val_cm[1,0]}, TP(Real)={val_cm[1,1]}")

# 8. OFFICIAL TEST EVALUATION (EVALUATED EXACTLY ONCE)
print("\n" + "=" * 70)
print("FINAL MODEL — OFFICIAL TEST BENCHMARK (N=518, EVALUATED ONCE)")
print("=" * 70)
with torch.no_grad():
    test_logits = []
    for X_tb, _ in test_loader:
        test_logits.append(model(X_tb.to(DEVICE)).cpu())
    test_logits = torch.cat(test_logits).numpy().reshape(-1, 1)

test_probs_calibrated = platt_calibrator.predict_proba(test_logits)[:, 1]
test_auc = roc_auc_score(y_test, test_probs_calibrated)
test_preds = (test_probs_calibrated >= best_thresh).astype(int)

test_acc = accuracy_score(y_test, test_preds)
test_bacc = balanced_accuracy_score(y_test, test_preds)
test_prec_real = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
test_rec_real = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
test_f1_real = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
test_prec_fake = precision_score(y_test, test_preds, pos_label=0, zero_division=0)
test_rec_fake = recall_score(y_test, test_preds, pos_label=0, zero_division=0)
test_f1_fake = f1_score(y_test, test_preds, pos_label=0, zero_division=0)
test_cm = confusion_matrix(y_test, test_preds)

print(f"Accuracy         : {test_acc*100:.2f}%")
print(f"Balanced Accuracy: {test_bacc*100:.2f}%")
print(f"ROC-AUC          : {test_auc*100:.2f}%")
print(f"Real Class (1)   : Precision={test_prec_real*100:.2f}%, Recall={test_rec_real*100:.2f}%, F1={test_f1_real*100:.2f}%")
print(f"Fake Class (0)   : Precision={test_prec_fake*100:.2f}%, Recall={test_rec_fake*100:.2f}%, F1={test_f1_fake*100:.2f}%")
print(f"Confusion Matrix : TN(Fake)={test_cm[0,0]}, FP(Fake->Real)={test_cm[0,1]}, FN(Real->Fake)={test_cm[1,0]}, TP(Real)={test_cm[1,1]}")

# 9. SAVE FINAL CONFIG & METRICS
config_artifact = {
    "model_type": "FaceSpatialTemporalModel_V3",
    "architecture": "Projection(512->256) + LayerNorm + 2-layer BiGRU(128) + MultiHeadAttention(4) + DualPooling + MLP",
    "feature_dim": 512,
    "hidden_dim": 128,
    "sequence_length": 8,
    "threshold": float(best_thresh),
    "label_mapping": {0: "Fake", 1: "Real"},
    "pos_weight": float(pos_weight_val),
    "best_epoch": int(best_epoch),
    "validation_auc": float(best_val_auc),
    "test_auc": float(test_auc)
}
with open(os.path.join(MODEL_FINAL_DIR, "video_config.pkl"), "wb") as f:
    pickle.dump(config_artifact, f)

metrics_artifact = {
    "validation": {
        "accuracy": float(val_acc),
        "balanced_accuracy": float(val_bacc),
        "roc_auc": float(best_val_auc),
        "real_precision": float(val_prec_real),
        "real_recall": float(val_rec_real),
        "real_f1": float(val_f1_real),
        "fake_precision": float(val_prec_fake),
        "fake_recall": float(val_rec_fake),
        "fake_f1": float(val_f1_fake),
        "confusion_matrix": val_cm.tolist()
    },
    "test": {
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bacc),
        "roc_auc": float(test_auc),
        "real_precision": float(test_prec_real),
        "real_recall": float(test_rec_real),
        "real_f1": float(test_f1_real),
        "fake_precision": float(test_prec_fake),
        "fake_recall": float(test_rec_fake),
        "fake_f1": float(test_f1_fake),
        "confusion_matrix": test_cm.tolist()
    }
}
with open(os.path.join(MODEL_FINAL_DIR, "video_metrics.pkl"), "wb") as f:
    pickle.dump(metrics_artifact, f)

print(f"\nALL ARTIFACTS SAVED TO {MODEL_FINAL_DIR}")
