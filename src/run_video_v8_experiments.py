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
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "models", "video_experiments_v8")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"VIDEO DETECTOR V8 FOCUSED EXPERIMENT ENGINE (Device: {DEVICE})")
print("=" * 80)

# ----------------------------------------------------------------------
# STEP 1 & 2: LOAD & STANDARDIZE FEATURE DATASET
# ----------------------------------------------------------------------
print("\nSTEP 1 & 2: LOADING FEATURE DATASET & VERIFYING SPLITS...")
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
face_det_all = df["face_detected_frames"].values.astype(np.float32)

X_train_raw = X_raw[train_mask]
y_train = y_all[train_mask]
face_det_train = face_det_all[train_mask]

X_val_raw = X_raw[val_mask]
y_val = y_all[val_mask]
face_det_val = face_det_all[val_mask]

X_test_raw = X_raw[test_mask]
y_test = y_all[test_mask]
face_det_test = face_det_all[test_mask]

# Feature Standardization strictly on Train split
train_flat = X_train_raw.reshape(-1, 512)
feat_mean = train_flat.mean(axis=0)
feat_std = train_flat.std(axis=0)
feat_std[feat_std < 1e-6] = 1.0

X_train = (X_train_raw - feat_mean) / feat_std
X_val = (X_val_raw - feat_mean) / feat_std
X_test = (X_test_raw - feat_mean) / feat_std

# ----------------------------------------------------------------------
# STEP 3: COMPUTE QUALITY METADATA DESCRIPTORS
# ----------------------------------------------------------------------
print("\nSTEP 3: COMPUTING QUALITY METADATA DESCRIPTORS...")

def extract_quality_descriptors(X_norm, face_counts):
    # X_norm: [N, 8, 512]
    # Quality features per frame:
    # 1. Feature L2 norm per frame (contrast/energy)
    # 2. Consecutive frame cosine similarity
    # 3. Normalized face detected ratio for the video
    N, T, D = X_norm.shape
    norms = np.linalg.norm(X_norm, axis=2, keepdims=True) # [N, 8, 1]
    
    # Cosine similarity
    unit_X = X_norm / (norms + 1e-8)
    cos_sim = np.ones((N, T, 1), dtype=np.float32)
    for t in range(1, T):
        sim = np.sum(unit_X[:, t, :] * unit_X[:, t-1, :], axis=1, keepdims=True)
        cos_sim[:, t, 0] = sim[:, 0]
        
    face_ratio = (face_counts / 8.0).reshape(N, 1, 1).repeat(T, axis=1) # [N, 8, 1]
    
    # Concatenate quality features [N, 8, 3]
    quality_feats = np.concatenate([norms / 25.0, cos_sim, face_ratio], axis=2)
    return quality_feats

Q_train = extract_quality_descriptors(X_train, face_det_train)
Q_val = extract_quality_descriptors(X_val, face_det_val)
Q_test = extract_quality_descriptors(X_test, face_det_test)

# Combined X + Q
XQ_train = np.concatenate([X_train, Q_train], axis=2) # [N, 8, 515]
XQ_val = np.concatenate([X_val, Q_val], axis=2)
XQ_test = np.concatenate([X_test, Q_test], axis=2)

print(f"  Standard Features: {X_train.shape}")
print(f"  Quality Descriptors: {Q_train.shape}")
print(f"  Quality-Concatenated: {XQ_train.shape}")

# Dataset Class supporting quality and random frame dropout
class VideoDatasetV8(Dataset):
    def __init__(self, X, y, Q=None, mask_prob=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.Q = torch.tensor(Q, dtype=torch.float32) if Q is not None else None
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.mask_prob > 0.0:
            for t in range(x.shape[0]):
                if random.random() < self.mask_prob:
                    x[t] = 0.0
        if self.Q is not None:
            q = self.Q[idx].clone()
            return x, q, y
        return x, y

# ----------------------------------------------------------------------
# STEP 4 & 5: MODEL ARCHITECTURES
# ----------------------------------------------------------------------

# 1. Candidate A / T1: V4 Control (Conv1D + BiGRU + MHA + Dual Pooling)
class TemporalConvBiGRUControl(nn.Module):
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

# 2. Candidate T2: Quality-Concatenated Temporal Model
class QualityConcatTemporalModel(nn.Module):
    def __init__(self, input_dim=515, hidden_dim=128, dropout=0.25):
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

# 3. Candidate T3: Quality-Gated Temporal Model
class QualityGatedTemporalModel(nn.Module):
    def __init__(self, input_dim=512, quality_dim=3, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.GELU()
        self.q_gate = nn.Sequential(
            nn.Linear(quality_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
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

    def forward(self, x, q):
        x_t = x.transpose(1, 2)
        c_out = self.act1(self.bn1(self.conv1(x_t))).transpose(1, 2)
        gate = self.q_gate(q) # [B, 8, 256]
        gated_c = c_out * gate
        gru_out, _ = self.gru(gated_c)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# 4. Candidate T4: Quality-Weighted Temporal Pooling Model
class QualityWeightedPoolingModel(nn.Module):
    def __init__(self, input_dim=512, quality_dim=3, hidden_dim=128, dropout=0.25):
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
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + quality_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x, q):
        x_t = x.transpose(1, 2)
        c_out = self.act1(self.bn1(self.conv1(x_t))).transpose(1, 2)
        gru_out, _ = self.gru(c_out)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        norm_out = self.norm(gru_out + attn_out)
        
        # Standard mean + max
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        
        # Quality-aware attention pooling
        pool_in = torch.cat([norm_out, q], dim=2) # [B, 8, 256 + 3]
        pool_weights = F.softmax(self.pool_gate(pool_in), dim=1) # [B, 8, 1]
        q_pool = (norm_out * pool_weights).sum(dim=1) # [B, 256]
        
        pooled = torch.cat([mean_p, max_p, q_pool], dim=1)
        return self.classifier(pooled).squeeze(1)

# ----------------------------------------------------------------------
# STEP 8: TRAINING & EVALUATION HARNESS
# ----------------------------------------------------------------------

# Focal Loss Implementation
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

def train_and_eval_v8_model(
    model,
    X_tr, y_tr,
    X_v, y_v,
    X_te, y_te,
    Q_tr=None, Q_v=None, Q_te=None,
    exp_name="Experiment",
    epochs=15,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-3,
    pos_weight=7.438,
    use_focal=False,
    mask_prob=0.0
):
    model = model.to(DEVICE)
    train_ds = VideoDatasetV8(X_tr, y_tr, Q=Q_tr, mask_prob=mask_prob)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_x_t = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    test_x_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    val_q_t = torch.tensor(Q_v, dtype=torch.float32).to(DEVICE) if Q_v is not None else None
    test_q_t = torch.tensor(Q_te, dtype=torch.float32).to(DEVICE) if Q_te is not None else None
    
    if use_focal:
        criterion = BinaryFocalLoss(alpha=0.75, gamma=1.5)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_score = -1.0
    best_state = None
    best_epoch = -1
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if Q_tr is not None:
                batch_x, batch_q, batch_y = batch
                batch_x, batch_q, batch_y = batch_x.to(DEVICE), batch_q.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch_x, batch_q)
            else:
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch_x)
                
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if Q_v is not None:
                v_logits = model(val_x_t, val_q_t).cpu().numpy()
            else:
                v_logits = model(val_x_t).cpu().numpy()
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
        if Q_v is not None:
            val_logits = model(val_x_t, val_q_t).cpu().numpy()
            test_logits = model(test_x_t, test_q_t).cpu().numpy()
        else:
            val_logits = model(val_x_t).cpu().numpy()
            test_logits = model(test_x_t).cpu().numpy()
            
    # Calibrate strictly on Validation Logits
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(val_logits.reshape(-1, 1), y_v)
    
    val_cal_probs = calibrator.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    test_cal_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    
    # Validation Threshold Search
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
                
    # Evaluate Validation Performance
    val_pred = (val_cal_probs >= best_t).astype(int)
    val_acc = accuracy_score(y_v, val_pred)
    val_bal_acc = balanced_accuracy_score(y_v, val_pred)
    val_auc = roc_auc_score(y_v, val_cal_probs)
    val_f1_real = f1_score(y_v, val_pred, pos_label=1, zero_division=0)
    val_rec_real = recall_score(y_v, val_pred, pos_label=1, zero_division=0)
    val_f1_fake = f1_score(y_v, val_pred, pos_label=0, zero_division=0)
    val_brier = brier_score_loss(y_v, val_cal_probs)
    
    # Evaluate Frozen Test Performance (once, strictly with val-selected threshold)
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
        "pos_weight": pos_weight if not use_focal else "Focal_1.5",
        "mask_prob": mask_prob,
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
# EXECUTE CONTROLLED V8 EXPERIMENT SUITE
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXECUTING CONTROLLED V8 EXPERIMENT SUITE")
print("=" * 80)

results = []

# 1. Candidate A / T1: V4 Control
print("\n--- 1. Candidate A: V4 Control (Conv1D + BiGRU + MHA) ---")
res_a = train_and_eval_v8_model(
    TemporalConvBiGRUControl(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V8_CandidateA_V4_Control",
    epochs=12, pos_weight=7.438
)
results.append(res_a)

# 2. Candidate T2: Quality-Concatenated Temporal Model
print("\n--- 2. Candidate T2: Quality-Concatenated Model (515-D) ---")
res_t2 = train_and_eval_v8_model(
    QualityConcatTemporalModel(input_dim=515, hidden_dim=128),
    XQ_train, y_train, XQ_val, y_val, XQ_test, y_test,
    exp_name="V8_CandidateT2_QualityConcat",
    epochs=12, pos_weight=7.438
)
results.append(res_t2)

# 3. Candidate T3: Quality-Gated Temporal Model
print("\n--- 3. Candidate T3: Quality-Gated Model ---")
res_t3 = train_and_eval_v8_model(
    QualityGatedTemporalModel(input_dim=512, quality_dim=3, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    Q_tr=Q_train, Q_v=Q_val, Q_te=Q_test,
    exp_name="V8_CandidateT3_QualityGated",
    epochs=15, pos_weight=7.438
)
results.append(res_t3)

# 4. Candidate T4: Quality-Weighted Pooling Model
print("\n--- 4. Candidate T4: Quality-Weighted Pooling Model ---")
res_t4 = train_and_eval_v8_model(
    QualityWeightedPoolingModel(input_dim=512, quality_dim=3, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    Q_tr=Q_train, Q_v=Q_val, Q_te=Q_test,
    exp_name="V8_CandidateT4_QualityWeightedPooling",
    epochs=15, pos_weight=7.438
)
results.append(res_t4)

# 5. Candidate R1: Robust Masking p=0.05
print("\n--- 5. Candidate R1: Robust Masking (p=0.05) ---")
res_r1 = train_and_eval_v8_model(
    TemporalConvBiGRUControl(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V8_CandidateR1_Masking_p0.05",
    epochs=15, pos_weight=7.438, mask_prob=0.05
)
results.append(res_r1)

# 6. Candidate R2: Robust Masking p=0.10
print("\n--- 6. Candidate R2: Robust Masking (p=0.10) ---")
res_r2 = train_and_eval_v8_model(
    TemporalConvBiGRUControl(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V8_CandidateR2_Masking_p0.10",
    epochs=15, pos_weight=7.438, mask_prob=0.10
)
results.append(res_r2)

# 7. Candidate L1: Moderate Class Weighting pos_w=5.5
print("\n--- 7. Candidate L1: Moderate Class Weighting (pos_weight=5.5) ---")
res_l1 = train_and_eval_v8_model(
    TemporalConvBiGRUControl(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V8_CandidateL1_ModerateWeight_5.5",
    epochs=15, pos_weight=5.50
)
results.append(res_l1)

# 8. Candidate L2: Binary Focal Loss (gamma=1.5)
print("\n--- 8. Candidate L2: Binary Focal Loss (gamma=1.5) ---")
res_l2 = train_and_eval_v8_model(
    TemporalConvBiGRUControl(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V8_CandidateL2_FocalLoss_gamma1.5",
    epochs=15, use_focal=True
)
results.append(res_l2)

# 9. Candidate F1: Probability Fusion
print("\n--- 9. Candidate F1: Probability Fusion (V4 Control + Best Quality Model) ---")
val_scores = [r["val_auc"] * 0.6 + r["val_bal_acc"] * 0.4 for r in results]
best_idx = int(np.argmax(val_scores))
best_cand = results[best_idx]
print(f"  Best Validation Candidate: {best_cand['experiment']} (Score: {val_scores[best_idx]:.4f})")

v4_val_p = res_a["val_cal_probs"]
v4_test_p = res_a["test_cal_probs"]
best_val_p = best_cand["val_cal_probs"]
best_test_p = best_cand["test_cal_probs"]

best_alpha = 0.5
best_fusion_val_score = -1.0
for alpha in np.arange(0.1, 1.0, 0.1):
    fused_val_p = alpha * v4_val_p + (1.0 - alpha) * best_val_p
    auc = roc_auc_score(y_val, fused_val_p)
    bal = balanced_accuracy_score(y_val, (fused_val_p >= 0.5).astype(int))
    sc = auc * 0.6 + bal * 0.4
    if sc > best_fusion_val_score:
        best_fusion_val_score = sc
        best_alpha = alpha

fused_val_p = best_alpha * v4_val_p + (1.0 - best_alpha) * best_val_p
fused_test_p = best_alpha * v4_test_p + (1.0 - best_alpha) * best_test_p

best_fusion_t = 0.5
best_sc = -1.0
for t in np.arange(0.10, 0.90, 0.01):
    pred = (fused_val_p >= t).astype(int)
    f1_r = f1_score(y_val, pred, pos_label=1, zero_division=0)
    f1_f = f1_score(y_val, pred, pos_label=0, zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, pred)
    rec_f = recall_score(y_val, pred, pos_label=0, zero_division=0)
    if rec_f >= 0.85:
        score = (f1_r + f1_f) / 2.0 + bal_acc * 0.5
        if score > best_sc:
            best_sc = score
            best_fusion_t = t

test_pred_fused = (fused_test_p >= best_fusion_t).astype(int)
test_acc_f = accuracy_score(y_test, test_pred_fused)
test_bal_acc_f = balanced_accuracy_score(y_test, test_pred_fused)
test_auc_f = roc_auc_score(y_test, fused_test_p)
test_prec_real_f = precision_score(y_test, test_pred_fused, pos_label=1, zero_division=0)
test_rec_real_f = recall_score(y_test, test_pred_fused, pos_label=1, zero_division=0)
test_f1_real_f = f1_score(y_test, test_pred_fused, pos_label=1, zero_division=0)
test_prec_fake_f = precision_score(y_test, test_pred_fused, pos_label=0, zero_division=0)
test_rec_fake_f = recall_score(y_test, test_pred_fused, pos_label=0, zero_division=0)
test_f1_fake_f = f1_score(y_test, test_pred_fused, pos_label=0, zero_division=0)
test_brier_f = brier_score_loss(y_test, fused_test_p)
cm_f = confusion_matrix(y_test, test_pred_fused)
tn_f, fp_f, fn_f, tp_f = cm_f.ravel()

res_fusion = {
    "experiment": f"V8_Fusion_V4_plus_{best_cand['experiment']}_(alpha={best_alpha:.1f})",
    "model_class": "ProbabilityEnsembleFusion",
    "pos_weight": 7.438,
    "mask_prob": 0.0,
    "best_epoch": 0,
    "threshold": round(best_fusion_t, 4),
    "val_auc": round(roc_auc_score(y_val, fused_val_p), 4),
    "val_acc": round(accuracy_score(y_val, (fused_val_p >= best_fusion_t).astype(int)), 4),
    "val_bal_acc": round(balanced_accuracy_score(y_val, (fused_val_p >= best_fusion_t).astype(int)), 4),
    "val_rec_real": round(recall_score(y_val, (fused_val_p >= best_fusion_t).astype(int), pos_label=1, zero_division=0), 4),
    "val_f1_real": round(f1_score(y_val, (fused_val_p >= best_fusion_t).astype(int), pos_label=1, zero_division=0), 4),
    "val_f1_fake": round(f1_score(y_val, (fused_val_p >= best_fusion_t).astype(int), pos_label=0, zero_division=0), 4),
    "val_brier": round(brier_score_loss(y_val, fused_val_p), 5),
    "test_auc": round(test_auc_f, 4),
    "test_acc": round(test_acc_f, 4),
    "test_bal_acc": round(test_bal_acc_f, 4),
    "test_prec_real": round(test_prec_real_f, 4),
    "test_rec_real": round(test_rec_real_f, 4),
    "test_f1_real": round(test_f1_real_f, 4),
    "test_prec_fake": round(test_prec_fake_f, 4),
    "test_rec_fake": round(test_rec_fake_f, 4),
    "test_f1_fake": round(test_f1_fake_f, 4),
    "test_brier": round(test_brier_f, 5),
    "TN": int(tn_f), "FP": int(fp_f), "FN": int(fn_f), "TP": int(tp_f),
    "total_errors": int(fp_f + fn_f)
}
results.append(res_fusion)

# ----------------------------------------------------------------------
# SAVE SUMMARY CSV REPORT
# ----------------------------------------------------------------------
summary_rows = []
csv_cols = [
    "experiment", "model_class", "pos_weight", "mask_prob", "best_epoch", "threshold",
    "val_auc", "val_acc", "val_bal_acc", "val_rec_real", "val_f1_real", "val_f1_fake", "val_brier",
    "test_auc", "test_acc", "test_bal_acc", "test_prec_real", "test_rec_real", "test_f1_real",
    "test_prec_fake", "test_rec_fake", "test_f1_fake", "test_brier", "TN", "FP", "FN", "TP", "total_errors"
]
for r in results:
    summary_rows.append({c: r[c] for c in csv_cols})

summary_df = pd.DataFrame(summary_rows)
csv_out = os.path.join(REPORTS_DIR, "video_v8_experiments_summary.csv")
summary_df.to_csv(csv_out, index=False)
print(f"\n[SAVED] Experiments summary saved to: {csv_out}")

# Print Table Summary
print("\n" + "=" * 105)
print(f"{'Experiment':<38} | {'Val AUC':<8} | {'Val BalAcc':<10} | {'Test Acc':<9} | {'Test BalAcc':<11} | {'Test AUC':<9} | {'Real F1':<8} | {'Errors':<6}")
print("-" * 105)
for r in results:
    print(f"{r['experiment']:<38} | {r['val_auc']*100:>6.2f}% | {r['val_bal_acc']*100:>8.2f}% | "
          f"{r['test_acc']*100:>7.2f}% | {r['test_bal_acc']*100:>9.2f}% | {r['test_auc']*100:>7.2f}% | "
          f"{r['test_f1_real']*100:>6.2f}% | {r['total_errors']:>6}")
print("=" * 105)

# ----------------------------------------------------------------------
# STEP 12: STRICT SELECTION RULE VERIFICATION
# ----------------------------------------------------------------------
print("\nSTEP 12: STRICT PRODUCTION WINNER SELECTION...")
v4_errors = res_a["total_errors"]
v4_bal_acc = res_a["test_bal_acc"]
v4_auc = res_a["test_auc"]

print(f"  V4 Control Baseline: Test Acc={res_a['test_acc']*100:.2f}%, BalAcc={v4_bal_acc*100:.2f}%, "
      f"ROC-AUC={v4_auc*100:.2f}%, Real F1={res_a['test_f1_real']*100:.2f}%, Errors={v4_errors}")

# Top V8 Candidate
v8_candidates = [r for r in results if "Control" not in r["experiment"] and "Fusion" not in r["experiment"]]
best_v8 = max(v8_candidates, key=lambda x: (x["val_auc"] * 0.6 + x["val_bal_acc"] * 0.4))

print(f"\n  Top V8 Candidate: {best_v8['experiment']}")
print(f"    Validation: AUC={best_v8['val_auc']*100:.2f}%, BalAcc={best_v8['val_bal_acc']*100:.2f}%, Real F1={best_v8['val_f1_real']*100:.2f}%")
print(f"    Test      : Acc={best_v8['test_acc']*100:.2f}%, BalAcc={best_v8['test_bal_acc']*100:.2f}%, ROC-AUC={best_v8['test_auc']*100:.2f}%, "
      f"Real F1={best_v8['test_f1_real']*100:.2f}%, Errors={best_v8['total_errors']}")

v8_is_winner = (
    best_v8["test_bal_acc"] > v4_bal_acc and
    best_v8["test_f1_real"] > res_a["test_f1_real"] and
    best_v8["total_errors"] <= v4_errors
)

print(f"\n  Selection Criteria Status: Genuine Win = {v8_is_winner}")

# Save V8 Artifacts
OUT_V8_DIR = os.path.join(BASE_DIR, "models", "video_final_v8")
os.makedirs(OUT_V8_DIR, exist_ok=True)

torch.save(best_v8["model_state"], os.path.join(OUT_V8_DIR, "temporal_attention_model.pth"))

with open(os.path.join(OUT_V8_DIR, "video_scaler.pkl"), "wb") as f:
    pickle.dump({"mean": feat_mean, "std": feat_std}, f)

with open(os.path.join(OUT_V8_DIR, "probability_calibrator.pkl"), "wb") as f:
    pickle.dump({"model": best_v8["calibrator"], "threshold": best_v8["threshold"]}, f)

with open(os.path.join(OUT_V8_DIR, "video_config.pkl"), "wb") as f:
    pickle.dump({
        "model_name": "VideoDetector_V8",
        "architecture": best_v8["model_class"],
        "experiment": best_v8["experiment"],
        "frames_sampled": 8,
        "face_feature_dim": 512,
        "quality_dim": 3 if "Quality" in best_v8["model_class"] else 0,
        "decision_threshold": best_v8["threshold"]
    }, f)

with open(os.path.join(OUT_V8_DIR, "video_metrics.pkl"), "wb") as f:
    pickle.dump(best_v8, f)

print(f"[SAVED] VideoDetector_V8 model and artifacts saved to: {OUT_V8_DIR}")
print("\nV8 EXPERIMENT CAMPAIGN COMPLETE.")
