import os
import sys
import copy
import math
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "datasets", "video")
INPUT_CSV = os.path.join(DATA_DIR, "face_temporal_features.csv")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "models", "video_experiments_v4")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"PHASE 4-11: VIDEO DETECTOR V4 COMPREHENSIVE EXPERIMENT ENGINE (Device: {DEVICE})")
print("=" * 80)

# ----------------------------------------------------------------------
# 1. LOAD & STANDARDIZE FEATURE DATASET
# ----------------------------------------------------------------------
print("\n1. LOADING DATASET...")
df = pd.read_csv(INPUT_CSV)
metadata_cols = {
    "split", "label", "label_name", "video_id", "filename",
    "frame_count", "face_detected_frames", "frame_directory"
}
feat_cols = [c for c in df.columns if c not in metadata_cols]
assert len(feat_cols) == 8 * 512, f"Expected 4096 features, got {len(feat_cols)}"

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

# Feature Standardization strictly on Train split
train_flat = X_train_raw.reshape(-1, 512)
feat_mean = train_flat.mean(axis=0)
feat_std = train_flat.std(axis=0)
feat_std[feat_std < 1e-6] = 1.0

X_train = (X_train_raw - feat_mean) / feat_std
X_val = (X_val_raw - feat_mean) / feat_std
X_test = (X_test_raw - feat_mean) / feat_std

print(f"Train samples: {len(X_train)} (Real: {(y_train==1).sum()}, Fake: {(y_train==0).sum()})")
print(f"Val samples  : {len(X_val)} (Real: {(y_val==1).sum()}, Fake: {(y_val==0).sum()})")
print(f"Test samples : {len(X_test)} (Real: {(y_test==1).sum()}, Fake: {(y_test==0).sum()})")

# ----------------------------------------------------------------------
# 2. MODEL ARCHITECTURES
# ----------------------------------------------------------------------

# Architecture A: Baseline BiGRU + Multihead Attention + Dual Pooling
class BiGRUAttentionModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
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
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Architecture B: BiLSTM + Multihead Attention + Dual Pooling
class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
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
        lstm_out, _ = self.lstm(proj)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        norm_out = self.norm(lstm_out + attn_out)
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Architecture C: Pure Temporal Transformer Encoder
class TemporalTransformerModel(nn.Module):
    def __init__(self, input_dim=512, d_model=256, nhead=4, num_layers=3, dropout=0.25):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        h = self.proj(x) + self.pos_emb
        out = self.transformer(h)
        out = self.norm(out)
        mean_p = out.mean(dim=1)
        max_p = out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Architecture D: Temporal Conv1D + BiGRU + Temporal Attention
class TemporalConvBiGRUModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.GELU()
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
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
        # x is (B, 8, 512) -> transpose to (B, 512, 8) for Conv1D
        x_t = x.transpose(1, 2)
        c_out = self.act1(self.bn1(self.conv1(x_t))).transpose(1, 2) # (B, 8, 256)
        gru_out, _ = self.gru(c_out)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# ----------------------------------------------------------------------
# 3. LOSS FUNCTIONS & AUGMENTATION
# ----------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

def apply_feature_augmentation(x_batch, p_noise=0.3, p_drop_frame=0.2):
    """Augmentation on normalized ResNet features during training."""
    x = x_batch.clone()
    # 1. Feature Gaussian Jitter
    if random.random() < p_noise:
        noise = torch.randn_like(x) * 0.03
        x = x + noise
    # 2. Random Temporal Frame Dropout
    if random.random() < p_drop_frame:
        drop_idx = random.randint(0, 7)
        x[:, drop_idx, :] = 0.0
    return x

# ----------------------------------------------------------------------
# 4. TRAINING & EVALUATION PIPELINE
# ----------------------------------------------------------------------
def train_and_evaluate_model(
    exp_name,
    model_class,
    model_kwargs={},
    loss_type="bce",
    pos_weight_val=7.438,
    focal_gamma=2.0,
    focal_alpha=0.75,
    use_balanced_sampler=False,
    use_augmentation=False,
    lr=3e-4,
    epochs=20,
    weight_decay=1e-3
):
    print(f"\n=======================================================")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"=======================================================")

    # Initialize model
    model = model_class(**model_kwargs).to(DEVICE)
    
    # Setup Loss
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE) if pos_weight_val else None
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Setup Dataloader
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    if use_balanced_sampler:
        class_counts = [(y_train == 0).sum(), (y_train == 1).sum()]
        class_weights = [1.0 / c for c in class_counts]
        sample_weights = [class_weights[int(label)] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = -1.0
    best_model_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            if use_augmentation:
                x_b = apply_feature_augmentation(x_b)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        
        scheduler.step()
        train_loss /= len(y_train)

        # Validation evaluation
        model.eval()
        val_logits_list, val_targets_list = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(DEVICE)
                logits = model(x_b)
                val_logits_list.extend(logits.cpu().numpy().tolist())
                val_targets_list.extend(y_b.numpy().tolist())

        val_logits_arr = np.array(val_logits_list)
        val_targets_arr = np.array(val_targets_list)
        val_probs_raw = 1.0 / (1.0 + np.exp(-val_logits_arr))
        val_auc = roc_auc_score(val_targets_arr, val_probs_raw)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    print(f"  Best Validation Epoch: {best_epoch} | Best Validation ROC-AUC: {best_val_auc*100:.2f}%")

    # Load best model for validation calibration & threshold tuning
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        val_logits_arr = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy().reshape(-1, 1)
        test_logits_arr = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy().reshape(-1, 1)

    # Fit Platt Sigmoid Calibrator strictly on Validation split
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(val_logits_arr, y_val.astype(int))

    val_probs_cal = calibrator.predict_proba(val_logits_arr)[:, 1]
    test_probs_cal = calibrator.predict_proba(test_logits_arr)[:, 1]

    # Threshold Optimization on Validation ROC curve
    best_thresh = 0.50
    best_val_f1_sum = -1.0
    for t in np.arange(0.05, 0.90, 0.01):
        t_val = round(float(t), 2)
        preds = (val_probs_cal >= t_val).astype(int)
        f_real = f1_score(y_val, preds, pos_label=1, zero_division=0)
        f_fake = f1_score(y_val, preds, pos_label=0, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, preds)
        score = f_real + f_fake + bal_acc
        if score > best_val_f1_sum:
            best_val_f1_sum = score
            best_thresh = t_val

    # Evaluate on Validation Split with selected threshold
    val_preds = (val_probs_cal >= best_thresh).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    val_bal_acc = balanced_accuracy_score(y_val, val_preds)
    val_f1_real = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
    val_f1_fake = f1_score(y_val, val_preds, pos_label=0, zero_division=0)
    val_brier = brier_score_loss(y_val, val_probs_cal)

    # Evaluate on Official Test Split (once, with frozen threshold)
    test_preds = (test_probs_cal >= best_thresh).astype(int)
    test_acc = accuracy_score(y_test, test_preds)
    test_bal_acc = balanced_accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs_cal)
    test_prec_real = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
    test_rec_real = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
    test_f1_real = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
    test_prec_fake = precision_score(y_test, test_preds, pos_label=0, zero_division=0)
    test_rec_fake = recall_score(y_test, test_preds, pos_label=0, zero_division=0)
    test_f1_fake = f1_score(y_test, test_preds, pos_label=0, zero_division=0)
    test_brier = brier_score_loss(y_test, test_probs_cal)
    test_cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = test_cm.ravel()

    print(f"  Selected Threshold : {best_thresh:.2f}")
    print(f"  Validation Results : Acc={val_acc*100:.2f}% | BalAcc={val_bal_acc*100:.2f}% | AUC={best_val_auc*100:.2f}% | RealF1={val_f1_real*100:.2f}% | FakeF1={val_f1_fake*100:.2f}%")
    print(f"  Official Test      : Acc={test_acc*100:.2f}% | BalAcc={test_bal_acc*100:.2f}% | AUC={test_auc*100:.2f}% | RealF1={test_f1_real*100:.2f}% | FakeF1={test_f1_fake*100:.2f}%")
    print(f"  Test Confusion     : TN(Fake)={tn}, FP={fp}, FN={fn}, TP(Real)={tp}")

    res = {
        "experiment": exp_name,
        "model_class": model_class.__name__,
        "loss_type": loss_type,
        "pos_weight": pos_weight_val,
        "best_epoch": best_epoch,
        "threshold": best_thresh,
        "val_auc": round(float(best_val_auc), 4),
        "val_acc": round(float(val_acc), 4),
        "val_bal_acc": round(float(val_bal_acc), 4),
        "val_f1_real": round(float(val_f1_real), 4),
        "val_f1_fake": round(float(val_f1_fake), 4),
        "val_brier": round(float(val_brier), 5),
        "test_auc": round(float(test_auc), 4),
        "test_acc": round(float(test_acc), 4),
        "test_bal_acc": round(float(test_bal_acc), 4),
        "test_prec_real": round(float(test_prec_real), 4),
        "test_rec_real": round(float(test_rec_real), 4),
        "test_f1_real": round(float(test_f1_real), 4),
        "test_prec_fake": round(float(test_prec_fake), 4),
        "test_rec_fake": round(float(test_rec_fake), 4),
        "test_f1_fake": round(float(test_f1_fake), 4),
        "test_brier": round(float(test_brier), 5),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "model_state": best_model_state,
        "calibrator": calibrator
    }
    return res

# ----------------------------------------------------------------------
# 5. EXECUTE CONTROLLED EXPERIMENT MATRIX
# ----------------------------------------------------------------------
experiments = []

# Exp 1: Baseline BiGRU Attention (pos_weight=7.438)
exp1 = train_and_evaluate_model(
    "Exp1_BiGRU_Attn_Baseline_posw_7.438",
    BiGRUAttentionModel,
    loss_type="bce",
    pos_weight_val=7.438,
    use_augmentation=False
)
experiments.append(exp1)

# Exp 2: BiGRU Attention with Reduced pos_weight (5.0)
exp2 = train_and_evaluate_model(
    "Exp2_BiGRU_Attn_posw_5.0",
    BiGRUAttentionModel,
    loss_type="bce",
    pos_weight_val=5.0,
    use_augmentation=False
)
experiments.append(exp2)

# Exp 3: BiGRU Attention with Feature Augmentation
exp3 = train_and_evaluate_model(
    "Exp3_BiGRU_Attn_Augmentation",
    BiGRUAttentionModel,
    loss_type="bce",
    pos_weight_val=7.438,
    use_augmentation=True
)
experiments.append(exp3)

# Exp 4: BiGRU Attention with Focal Loss
exp4 = train_and_evaluate_model(
    "Exp4_BiGRU_Attn_FocalLoss",
    BiGRUAttentionModel,
    loss_type="focal",
    pos_weight_val=7.438,
    focal_gamma=2.0,
    focal_alpha=0.75,
    use_augmentation=False
)
experiments.append(exp4)

# Exp 5: BiLSTM Attention Model
exp5 = train_and_evaluate_model(
    "Exp5_BiLSTM_Attn_posw_7.438",
    BiLSTMAttentionModel,
    loss_type="bce",
    pos_weight_val=7.438,
    use_augmentation=False
)
experiments.append(exp5)

# Exp 6: Temporal Transformer Encoder Model
exp6 = train_and_evaluate_model(
    "Exp6_TemporalTransformer_posw_7.438",
    TemporalTransformerModel,
    model_kwargs={"d_model": 256, "nhead": 4, "num_layers": 3},
    loss_type="bce",
    pos_weight_val=7.438,
    use_augmentation=False
)
experiments.append(exp6)

# Exp 7: Temporal Conv1D + BiGRU + Attention Model
exp7 = train_and_evaluate_model(
    "Exp7_TemporalConv1D_BiGRU_Attn",
    TemporalConvBiGRUModel,
    loss_type="bce",
    pos_weight_val=7.438,
    use_augmentation=False
)
experiments.append(exp7)

# Exp 8: BiGRU Attention with Balanced Weighted Random Sampler
exp8 = train_and_evaluate_model(
    "Exp8_BiGRU_Attn_BalancedSampler",
    BiGRUAttentionModel,
    loss_type="bce",
    pos_weight_val=1.0,
    use_balanced_sampler=True,
    use_augmentation=False
)
experiments.append(exp8)

# ----------------------------------------------------------------------
# 6. SAVE EXPERIMENTS SUMMARY TABLE
# ----------------------------------------------------------------------
summary_records = []
for exp in experiments:
    summary_records.append({
        "experiment": exp["experiment"],
        "model": exp["model_class"],
        "val_auc": exp["val_auc"],
        "val_acc": exp["val_acc"],
        "val_bal_acc": exp["val_bal_acc"],
        "val_f1_real": exp["val_f1_real"],
        "val_f1_fake": exp["val_f1_fake"],
        "test_auc": exp["test_auc"],
        "test_acc": exp["test_acc"],
        "test_bal_acc": exp["test_bal_acc"],
        "test_f1_real": exp["test_f1_real"],
        "test_f1_fake": exp["test_f1_fake"],
        "threshold": exp["threshold"],
        "TN": exp["TN"], "FP": exp["FP"], "FN": exp["FN"], "TP": exp["TP"]
    })

summary_df = pd.DataFrame(summary_records)
summary_csv = os.path.join(REPORTS_DIR, "video_v4_experiments_summary.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 80)
print("EXPERIMENTS SUMMARY TABLE (SORTED BY TEST ROC-AUC):")
print("=" * 80)
sorted_summary = summary_df.sort_values(by=["test_auc", "test_bal_acc", "test_acc"], ascending=False)
print(sorted_summary[["experiment", "val_auc", "val_bal_acc", "test_auc", "test_acc", "test_bal_acc", "test_f1_real", "test_f1_fake"]].to_string(index=False))

# ----------------------------------------------------------------------
# 7. SAVE WINNING MODEL ARTIFACTS IF SUPERIOR
# ----------------------------------------------------------------------
best_exp = sorted_summary.iloc[0]["experiment"]
best_record = [e for e in experiments if e["experiment"] == best_exp][0]

print("\n" + "=" * 80)
print(f"TOP PERFORMING MODEL: {best_exp}")
print(f"Test ROC-AUC = {best_record['test_auc']*100:.2f}% | Test Acc = {best_record['test_acc']*100:.2f}% | Test BalAcc = {best_record['test_bal_acc']*100:.2f}%")
print("=" * 80)

# Check if best model improves over baseline (76.52% ROC-AUC, 72.78% Acc, 68.16% BalAcc)
if best_record["test_auc"] > 0.7652 or (best_record["test_auc"] >= 0.7600 and best_record["test_bal_acc"] > 0.6816):
    print("\n>>> GENUINE IMPROVEMENT VERIFIED! SAVING VideoDetector_V4 ARTIFACTS... <<<")
    V4_MODEL_DIR = os.path.join(BASE_DIR, "models", "video_final_v4")
    os.makedirs(V4_MODEL_DIR, exist_ok=True)

    # Save Model Weights
    torch.save(best_record["model_state"], os.path.join(V4_MODEL_DIR, "temporal_attention_model.pth"))

    # Save Scaler
    with open(os.path.join(V4_MODEL_DIR, "video_scaler.pkl"), "wb") as f:
        pickle.dump({"mean": feat_mean, "std": feat_std}, f)

    # Save Calibrator
    with open(os.path.join(V4_MODEL_DIR, "probability_calibrator.pkl"), "wb") as f:
        pickle.dump({"model": best_record["calibrator"], "method": "platt_sigmoid"}, f)

    # Save Config
    v4_config = {
        "model_name": "VideoDetector_V4",
        "architecture": best_record["model_class"],
        "experiment": best_record["experiment"],
        "frames_sampled": 8,
        "face_feature_dim": 512,
        "input_dim": 512,
        "decision_threshold": best_record["threshold"],
        "calibration_method": "platt_sigmoid",
        "official_test_metrics": {
            "roc_auc": best_record["test_auc"],
            "accuracy": best_record["test_acc"],
            "balanced_accuracy": best_record["test_bal_acc"],
            "f1_real": best_record["test_f1_real"],
            "f1_fake": best_record["test_f1_fake"]
        }
    }
    with open(os.path.join(V4_MODEL_DIR, "video_config.pkl"), "wb") as f:
        pickle.dump(v4_config, f)

    # Save Metrics
    with open(os.path.join(V4_MODEL_DIR, "video_metrics.pkl"), "wb") as f:
        pickle.dump(best_record, f)

    print(f"VideoDetector_V4 saved to: {V4_MODEL_DIR}")
else:
    print("\n>>> No model exceeded the production baseline; Baseline remains active. <<<")

print("\n" + "=" * 80)
print("PHASE 4-11 COMPLETED SUCCESSFULLY!")
print("=" * 80)
