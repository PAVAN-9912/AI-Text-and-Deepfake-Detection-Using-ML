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
from torch.utils.data import DataLoader, Dataset
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
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "models", "video_experiments_v9")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"VIDEO DETECTOR V9 TARGETED FRAME-MASKING EXPERIMENT ENGINE (Device: {DEVICE})")
print("=" * 80)

# ----------------------------------------------------------------------
# STEP 1: LOAD & STANDARDIZE FEATURE DATASET
# ----------------------------------------------------------------------
print("\nSTEP 1: LOADING FEATURE DATASET & VERIFYING SPLITS...")
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

train_ids = set(df.loc[train_mask, "video_id"].unique())
val_ids = set(df.loc[val_mask, "video_id"].unique())
test_ids = set(df.loc[test_mask, "video_id"].unique())

assert len(train_ids.intersection(val_ids)) == 0, "Leakage: Train & Val overlap!"
assert len(train_ids.intersection(test_ids)) == 0, "Leakage: Train & Test overlap!"
assert len(val_ids.intersection(test_ids)) == 0, "Leakage: Val & Test overlap!"

print(f"  Train samples     : {train_mask.sum()} (Real: {(df.loc[train_mask, 'label']==1).sum()}, Fake: {(df.loc[train_mask, 'label']==0).sum()})")
print(f"  Validation samples: {val_mask.sum()} (Real: {(df.loc[val_mask, 'label']==1).sum()}, Fake: {(df.loc[val_mask, 'label']==0).sum()})")
print(f"  Frozen Test samples: {test_mask.sum()} (Real: {(df.loc[test_mask, 'label']==1).sum()}, Fake: {(df.loc[test_mask, 'label']==0).sum()})")

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

# ----------------------------------------------------------------------
# DATASET WITH ADVANCED TRAINING-TIME MASKING & JITTER STRATEGIES
# ----------------------------------------------------------------------
class AdvancedMaskedVideoDataset(Dataset):
    def __init__(self, X, y, mask_strategy="none", mask_prob=0.0, jitter_std=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask_strategy = mask_strategy
        self.mask_prob = mask_prob
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        T, D = x.shape
        
        # 1. Random independent frame masking
        if self.mask_strategy == "independent" and self.mask_prob > 0.0:
            for t in range(T):
                if random.random() < self.mask_prob:
                    x[t] = 0.0
                    
        # 2. Contiguous block frame masking
        elif self.mask_strategy == "contiguous" and self.mask_prob > 0.0:
            if random.random() < self.mask_prob:
                block_len = random.choice([1, 2])
                start_idx = random.randint(0, max(0, T - block_len))
                x[start_idx:start_idx + block_len] = 0.0
                
        # 3. Energy / norm aware masking
        elif self.mask_strategy == "energy_aware" and self.mask_prob > 0.0:
            norms = torch.norm(x, dim=1) # [T]
            min_norm_idx = torch.argmin(norms).item()
            if random.random() < self.mask_prob:
                x[min_norm_idx] = 0.0
                
        # 4. Temporal jitter
        if self.jitter_std > 0.0:
            noise = torch.randn_like(x) * self.jitter_std
            x = x + noise
            
        return x, y

# ----------------------------------------------------------------------
# V4 TEMPORAL CONV-BIGRU ARCHITECTURE
# ----------------------------------------------------------------------
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
        x_t = x.transpose(1, 2)
        c_out = self.act1(self.bn1(self.conv1(x_t))).transpose(1, 2)
        gru_out, _ = self.gru(c_out)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Binary Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return focal_loss.mean()

# ----------------------------------------------------------------------
# TRAINING & EVALUATION HARNESS
# ----------------------------------------------------------------------
def train_and_eval_v9_model(
    model,
    X_tr, y_tr,
    X_v, y_v,
    X_te, y_te,
    exp_name="Experiment",
    mask_strategy="none",
    mask_prob=0.0,
    jitter_std=0.0,
    epochs=15,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-3,
    pos_weight=7.438,
    loss_type="bce",
    gamma=1.5
):
    model = model.to(DEVICE)
    train_ds = AdvancedMaskedVideoDataset(
        X_tr, y_tr,
        mask_strategy=mask_strategy,
        mask_prob=mask_prob,
        jitter_std=jitter_std
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_tensor = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    test_tensor = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    
    if loss_type == "focal":
        criterion = BinaryFocalLoss(alpha=0.75, gamma=gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_score = -1.0
    best_state = None
    best_epoch = -1
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            v_logits = model(val_tensor).cpu().numpy()
            v_probs = 1.0 / (1.0 + np.exp(-v_logits))
            
        v_auc = roc_auc_score(y_v, v_probs)
        v_bal_acc = balanced_accuracy_score(y_v, (v_probs >= 0.5).astype(int))
        v_score = v_auc * 0.6 + v_bal_acc * 0.4
        
        if v_score > best_val_score:
            best_val_score = v_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        val_logits = model(val_tensor).cpu().numpy()
        test_logits = model(test_tensor).cpu().numpy()
        
    # Calibrate strictly on validation logits
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(val_logits.reshape(-1, 1), y_v)
    
    val_cal_probs = calibrator.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    test_cal_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    
    # Threshold search strictly on validation set
    best_t = 0.5
    best_score = -1.0
    for t in np.arange(0.10, 0.90, 0.01):
        pred = (val_cal_probs >= t).astype(int)
        f1_r = f1_score(y_v, pred, pos_label=1, zero_division=0)
        f1_f = f1_score(y_v, pred, pos_label=0, zero_division=0)
        bal_acc = balanced_accuracy_score(y_v, pred)
        rec_f = recall_score(y_v, pred, pos_label=0, zero_division=0)
        if rec_f >= 0.85:
            sc = (f1_r + f1_f) / 2.0 + bal_acc * 0.5
            if sc > best_score:
                best_score = sc
                best_t = t
                
    # Validation metrics
    val_pred = (val_cal_probs >= best_t).astype(int)
    val_acc = accuracy_score(y_v, val_pred)
    val_bal_acc = balanced_accuracy_score(y_v, val_pred)
    val_auc = roc_auc_score(y_v, val_cal_probs)
    val_f1_real = f1_score(y_v, val_pred, pos_label=1, zero_division=0)
    val_rec_real = recall_score(y_v, val_pred, pos_label=1, zero_division=0)
    val_f1_fake = f1_score(y_v, val_pred, pos_label=0, zero_division=0)
    val_brier = brier_score_loss(y_v, val_cal_probs)
    
    # Frozen test metrics (evaluated once strictly with val threshold)
    test_pred = (test_cal_probs >= best_t).astype(int)
    test_acc = accuracy_score(y_te, test_pred)
    test_bal_acc = balanced_accuracy_score(y_te, test_pred)
    test_auc = roc_auc_score(y_te, test_cal_probs)
    test_prec_real = precision_score(y_te, test_pred, pos_label=1, zero_division=0)
    test_rec_real = recall_score(y_te, test_pred, pos_label=1, zero_division=0)
    test_f1_real = f1_score(y_te, test_pred, pos_label=1, zero_division=0)
    test_prec_fake = precision_score(y_te, test_pred, pos_label=0, zero_division=0)
    test_rec_fake = recall_score(y_te, test_pred, pos_label=0, zero_division=0)
    test_f1_fake = f1_score(y_te, test_pred, pos_label=0, zero_division=0)
    test_brier = brier_score_loss(y_te, test_cal_probs)
    cm = confusion_matrix(y_te, test_pred)
    tn, fp, fn, tp = cm.ravel()
    total_errors = fp + fn
    
    res = {
        "experiment": exp_name,
        "model_class": model.__class__.__name__,
        "mask_strategy": mask_strategy,
        "mask_prob": mask_prob,
        "jitter_std": jitter_std,
        "pos_weight": pos_weight if loss_type == "bce" else f"Focal_{gamma}",
        "best_epoch": best_epoch,
        "threshold": round(best_t, 4),
        "val_auc": round(val_auc, 4),
        "val_acc": round(val_acc, 4),
        "val_bal_acc": round(val_bal_acc, 4),
        "val_rec_real": round(val_rec_real, 4),
        "val_f1_real": round(val_f1_real, 4),
        "val_f1_fake": round(val_f1_fake, 4),
        "val_brier": round(val_brier, 5),
        "test_auc": round(test_auc, 4),
        "test_acc": round(test_acc, 4),
        "test_bal_acc": round(test_bal_acc, 4),
        "test_prec_real": round(test_prec_real, 4),
        "test_rec_real": round(test_rec_real, 4),
        "test_f1_real": round(test_f1_real, 4),
        "test_prec_fake": round(test_prec_fake, 4),
        "test_rec_fake": round(test_rec_fake, 4),
        "test_f1_fake": round(test_f1_fake, 4),
        "test_brier": round(test_brier, 5),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "total_errors": int(total_errors),
        "val_cal_probs": val_cal_probs,
        "test_cal_probs": test_cal_probs,
        "calibrator": calibrator,
        "model_state": best_state
    }
    
    print(f"[{exp_name}] Val AUC: {val_auc*100:.2f}% | Val BalAcc: {val_bal_acc*100:.2f}% | "
          f"Test Acc: {test_acc*100:.2f}% | Test BalAcc: {test_bal_acc*100:.2f}% | "
          f"Test AUC: {test_auc*100:.2f}% | Real F1: {test_f1_real*100:.2f}% | Total Errors: {total_errors}")
    return res

# ----------------------------------------------------------------------
# EXECUTE SYSTEMATIC V9 EXPERIMENT SUITE
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXECUTING SYSTEMATIC V9 FRAME-MASKING & LOSS CAMPAIGN")
print("=" * 80)

results = []

# 1. Candidate A: V4 Baseline Control (No masking)
print("\n--- 1. Candidate A: V4 Control (p=0.0) ---")
res_a = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateA_V4_Control",
    mask_strategy="none", mask_prob=0.0, epochs=12, pos_weight=7.438
)
results.append(res_a)

# 2. Candidate B1: Independent Masking p=0.05
print("\n--- 2. Candidate B1: Independent Masking (p=0.05) ---")
res_b1 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateB1_Masking_p0.05",
    mask_strategy="independent", mask_prob=0.05, epochs=15, pos_weight=7.438
)
results.append(res_b1)

# 3. Candidate B2: Independent Masking p=0.08
print("\n--- 3. Candidate B2: Independent Masking (p=0.08) ---")
res_b2 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateB2_Masking_p0.08",
    mask_strategy="independent", mask_prob=0.08, epochs=15, pos_weight=7.438
)
results.append(res_b2)

# 4. Candidate B3: Independent Masking p=0.10
print("\n--- 4. Candidate B3: Independent Masking (p=0.10) ---")
res_b3 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateB3_Masking_p0.10",
    mask_strategy="independent", mask_prob=0.10, epochs=15, pos_weight=7.438
)
results.append(res_b3)

# 5. Candidate B4: Independent Masking p=0.12
print("\n--- 5. Candidate B4: Independent Masking (p=0.12) ---")
res_b4 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateB4_Masking_p0.12",
    mask_strategy="independent", mask_prob=0.12, epochs=15, pos_weight=7.438
)
results.append(res_b4)

# 6. Candidate B5: Independent Masking p=0.15
print("\n--- 6. Candidate B5: Independent Masking (p=0.15) ---")
res_b5 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateB5_Masking_p0.15",
    mask_strategy="independent", mask_prob=0.15, epochs=15, pos_weight=7.438
)
results.append(res_b5)

# 7. Candidate C1: Contiguous Block Masking (p=0.20)
print("\n--- 7. Candidate C1: Contiguous Block Masking (p=0.20) ---")
res_c1 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateC1_ContiguousMasking_p0.20",
    mask_strategy="contiguous", mask_prob=0.20, epochs=15, pos_weight=7.438
)
results.append(res_c1)

# 8. Candidate D1: Energy/Norm-Aware Masking (p=0.25)
print("\n--- 8. Candidate D1: Energy-Aware Masking (p=0.25) ---")
res_d1 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateD1_EnergyAwareMasking_p0.25",
    mask_strategy="energy_aware", mask_prob=0.25, epochs=15, pos_weight=7.438
)
results.append(res_d1)

# 9. Candidate E1: Masking (p=0.10) + Temporal Jitter (std=0.05)
print("\n--- 9. Candidate E1: Masking (p=0.10) + Jitter (std=0.05) ---")
res_e1 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateE1_Masking_p0.10_Jitter",
    mask_strategy="independent", mask_prob=0.10, jitter_std=0.05, epochs=15, pos_weight=7.438
)
results.append(res_e1)

# 10. Candidate L1: Masking (p=0.10) + Pos Weight=6.0
print("\n--- 10. Candidate L1: Masking (p=0.10) + Pos Weight=6.0 ---")
res_l1 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateL1_Masking_p0.10_PosW6.0",
    mask_strategy="independent", mask_prob=0.10, epochs=15, pos_weight=6.00
)
results.append(res_l1)

# 11. Candidate L2: Masking (p=0.10) + Focal Loss (gamma=1.5)
print("\n--- 11. Candidate L2: Masking (p=0.10) + Focal Loss (gamma=1.5) ---")
res_l2 = train_and_eval_v9_model(
    TemporalConvBiGRUModel(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V9_CandidateL2_Masking_p0.10_Focal1.5",
    mask_strategy="independent", mask_prob=0.10, epochs=15, loss_type="focal", gamma=1.5
)
results.append(res_l2)

# ----------------------------------------------------------------------
# SAVE SUMMARY CSV REPORT
# ----------------------------------------------------------------------
summary_rows = []
csv_cols = [
    "experiment", "model_class", "mask_strategy", "mask_prob", "jitter_std", "pos_weight", "best_epoch", "threshold",
    "val_auc", "val_acc", "val_bal_acc", "val_rec_real", "val_f1_real", "val_f1_fake", "val_brier",
    "test_auc", "test_acc", "test_bal_acc", "test_prec_real", "test_rec_real", "test_f1_real",
    "test_prec_fake", "test_rec_fake", "test_f1_fake", "test_brier", "TN", "FP", "FN", "TP", "total_errors"
]
for r in results:
    summary_rows.append({c: r[c] for c in csv_cols})

summary_df = pd.DataFrame(summary_rows)
csv_out = os.path.join(REPORTS_DIR, "video_v9_experiments_summary.csv")
summary_df.to_csv(csv_out, index=False)
print(f"\n[SAVED] Experiments summary saved to: {csv_out}")

# Print Table Summary
print("\n" + "=" * 110)
print(f"{'Experiment':<42} | {'Val AUC':<8} | {'Val BalAcc':<10} | {'Test Acc':<9} | {'Test BalAcc':<11} | {'Test AUC':<9} | {'Real F1':<8} | {'Errors':<6}")
print("-" * 110)
for r in results:
    print(f"{r['experiment']:<42} | {r['val_auc']*100:>6.2f}% | {r['val_bal_acc']*100:>8.2f}% | "
          f"{r['test_acc']*100:>7.2f}% | {r['test_bal_acc']*100:>9.2f}% | {r['test_auc']*100:>7.2f}% | "
          f"{r['test_f1_real']*100:>6.2f}% | {r['total_errors']:>6}")
print("=" * 110)

# ----------------------------------------------------------------------
# STEP 11: STRICT PRODUCTION WINNER SELECTION
# ----------------------------------------------------------------------
print("\nSTEP 11: STRICT PRODUCTION WINNER SELECTION...")
v4_errors = res_a["total_errors"]
v4_bal_acc = res_a["test_bal_acc"]
v4_auc = res_a["test_auc"]

print(f"  V4 Control Baseline: Test Acc={res_a['test_acc']*100:.2f}%, BalAcc={v4_bal_acc*100:.2f}%, "
      f"ROC-AUC={v4_auc*100:.2f}%, Real F1={res_a['test_f1_real']*100:.2f}%, Errors={v4_errors}")

# Top V9 Candidate selected strictly on Validation Set
v9_candidates = [r for r in results if "Control" not in r["experiment"]]
best_v9 = max(v9_candidates, key=lambda x: (x["val_auc"] * 0.6 + x["val_bal_acc"] * 0.4))

print(f"\n  Top V9 Candidate on Validation: {best_v9['experiment']}")
print(f"    Validation: AUC={best_v9['val_auc']*100:.2f}%, BalAcc={best_v9['val_bal_acc']*100:.2f}%, Real F1={best_v9['val_f1_real']*100:.2f}%")
print(f"    Test      : Acc={best_v9['test_acc']*100:.2f}%, BalAcc={best_v9['test_bal_acc']*100:.2f}%, ROC-AUC={best_v9['test_auc']*100:.2f}%, "
      f"Real F1={best_v9['test_f1_real']*100:.2f}%, Errors={best_v9['total_errors']}")

v9_is_winner = (
    best_v9["test_bal_acc"] > v4_bal_acc and
    best_v9["test_f1_real"] > res_a["test_f1_real"] and
    best_v9["total_errors"] <= v4_errors
)

print(f"\n  Selection Criteria Status: Genuine Win = {v9_is_winner}")

# Save V9 Artifacts
OUT_V9_DIR = os.path.join(BASE_DIR, "models", "video_final_v9")
os.makedirs(OUT_V9_DIR, exist_ok=True)

torch.save(best_v9["model_state"], os.path.join(OUT_V9_DIR, "temporal_attention_model.pth"))

with open(os.path.join(OUT_V9_DIR, "video_scaler.pkl"), "wb") as f:
    pickle.dump({"mean": feat_mean, "std": feat_std}, f)

with open(os.path.join(OUT_V9_DIR, "probability_calibrator.pkl"), "wb") as f:
    pickle.dump({"model": best_v9["calibrator"], "threshold": best_v9["threshold"]}, f)

with open(os.path.join(OUT_V9_DIR, "video_config.pkl"), "wb") as f:
    pickle.dump({
        "model_name": "VideoDetector_V9",
        "architecture": best_v9["model_class"],
        "experiment": best_v9["experiment"],
        "frames_sampled": 8,
        "face_feature_dim": 512,
        "mask_strategy": best_v9["mask_strategy"],
        "mask_prob": best_v9["mask_prob"],
        "decision_threshold": best_v9["threshold"]
    }, f)

with open(os.path.join(OUT_V9_DIR, "video_metrics.pkl"), "wb") as f:
    pickle.dump(best_v9, f)

print(f"[SAVED] VideoDetector_V9 model and artifacts saved to: {OUT_V9_DIR}")
print("\nV9 EXPERIMENT CAMPAIGN COMPLETE.")
