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
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "models", "video_experiments_v7")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"PHASE 1-16: VIDEO DETECTOR V7 COMPREHENSIVE EXPERIMENT ENGINE (Device: {DEVICE})")
print("=" * 80)

# ----------------------------------------------------------------------
# STEP 1 & 2: LOAD DATASET & VERIFY SPLITS
# ----------------------------------------------------------------------
print("\nSTEP 2: VERIFYING SPLIT INDEPENDENCE & ZERO LEAKAGE...")
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

assert len(train_ids.intersection(val_ids)) == 0, "LEAKAGE: Train & Val overlap!"
assert len(train_ids.intersection(test_ids)) == 0, "LEAKAGE: Train & Test overlap!"
assert len(val_ids.intersection(test_ids)) == 0, "LEAKAGE: Val & Test overlap!"

print(f"  Train videos     : {len(train_ids)} (rows: {train_mask.sum()})")
print(f"  Validation videos: {len(val_ids)} (rows: {val_mask.sum()})")
print(f"  Frozen Test videos: {len(test_ids)} (rows: {test_mask.sum()})")
print("  [CONFIRMED] Zero video-level leakage between Train, Validation, and Test splits.")

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

print(f"\nTrain Distribution: Total={len(y_train)}, Real={(y_train==1).sum()}, Fake={(y_train==0).sum()}")
print(f"Val Distribution  : Total={len(y_val)}, Real={(y_val==1).sum()}, Fake={(y_val==0).sum()}")
print(f"Test Distribution : Total={len(y_test)}, Real={(y_test==1).sum()}, Fake={(y_test==0).sum()}")

# ----------------------------------------------------------------------
# STEP 3: CONSTRUCT TEMPORAL DIFFERENCE & VELOCITY REPRESENTATIONS
# ----------------------------------------------------------------------
print("\nSTEP 3: COMPUTING AUGMENTED TEMPORAL FEATURE REPRESENTATIONS...")

def compute_temporal_diffs(X_data):
    # X_data: [N, 8, 512]
    diffs = np.zeros_like(X_data)
    diffs[:, 1:, :] = X_data[:, 1:, :] - X_data[:, :-1, :]
    # Concatenate along feature dimension -> [N, 8, 1024]
    return np.concatenate([X_data, diffs], axis=2)

def compute_cosine_consistency(X_data):
    # X_data: [N, 8, 512]
    N, T, D = X_data.shape
    norms = np.linalg.norm(X_data, axis=2, keepdims=True) + 1e-8
    normed_X = X_data / norms
    cos_sim = np.ones((N, T, 1), dtype=np.float32)
    for t in range(1, T):
        sim = np.sum(normed_X[:, t, :] * normed_X[:, t-1, :], axis=1, keepdims=True)
        cos_sim[:, t, 0] = sim[:, 0]
    return cos_sim

X_train_diff = compute_temporal_diffs(X_train)
X_val_diff = compute_temporal_diffs(X_val)
X_test_diff = compute_temporal_diffs(X_test)

X_train_cos = compute_cosine_consistency(X_train)
X_val_cos = compute_cosine_consistency(X_val)
X_test_cos = compute_cosine_consistency(X_test)

X_train_all = np.concatenate([X_train_diff, X_train_cos], axis=2) # [N, 8, 1025]
X_val_all = np.concatenate([X_val_diff, X_val_cos], axis=2)
X_test_all = np.concatenate([X_test_diff, X_test_cos], axis=2)

print(f"  Standard Features Shape : {X_train.shape}")
print(f"  Difference Features Shape: {X_train_diff.shape}")
print(f"  Full Augmented Shape    : {X_train_all.shape}")

# Dataset with optional temporal frame masking (Step 10: missing face robustness)
class RobustVideoDataset(Dataset):
    def __init__(self, X, y, mask_prob=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx].clone()
        y_sample = self.y[idx]
        if self.mask_prob > 0.0:
            for t in range(x_sample.shape[0]):
                if random.random() < self.mask_prob:
                    x_sample[t] = 0.0
        return x_sample, y_sample

# ----------------------------------------------------------------------
# STEP 7: MODEL ARCHITECTURES
# ----------------------------------------------------------------------

# Architecture A: V4 Control (Conv1D + BiGRU + MHA + Residual)
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

# Architecture B: V4 + Temporal Differences
class TemporalDiffBiGRUModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, dropout=0.25):
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

# Architecture C: CNN + Transformer
class CNNTemporalTransformer(nn.Module):
    def __init__(self, input_dim=512, d_model=256, nhead=4, num_layers=2, dropout=0.20):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act1 = nn.GELU()
        self.pos_embedding = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
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
        x_t = x.transpose(1, 2)
        c_out = self.act1(self.bn1(self.conv1(x_t))).transpose(1, 2)
        h = c_out + self.pos_embedding
        out = self.transformer(h)
        out = self.norm(out)
        mean_p = out.mean(dim=1)
        max_p = out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Architecture D: BiGRU + Transformer
class BiGRUTransformerModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, nhead=4, num_layers=2, dropout=0.20):
        super().__init__()
        self.projection = nn.Linear(input_dim, 256)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        proj = F.gelu(self.projection(x))
        gru_out, _ = self.gru(proj)
        trans_out = self.transformer(gru_out)
        out = self.norm(gru_out + trans_out)
        mean_p = out.mean(dim=1)
        max_p = out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Architecture E: Hybrid V7 (Conv + BiGRU + MHA + Attention-Weighted Pooling)
class HybridV7Model(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, dropout=0.25):
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
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, 128), # Mean (256) + Max (256) + Attn (256) = 768
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
        
        # Dual Pooling
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        
        # Attention Pooling
        weights = F.softmax(self.pool_attn(norm_out), dim=1) # [B, 8, 1]
        attn_p = (norm_out * weights).sum(dim=1) # [B, 256]
        
        pooled = torch.cat([mean_p, max_p, attn_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Architecture F: Dual-Stream Temporal Model
class DualStreamTemporalModel(nn.Module):
    def __init__(self, spatial_dim=512, diff_dim=512, hidden_dim=64, dropout=0.25):
        super().__init__()
        # Stream 1: Spatial
        self.s_conv = nn.Conv1d(spatial_dim, 128, kernel_size=3, padding=1)
        self.s_bn = nn.BatchNorm1d(128)
        self.s_gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        
        # Stream 2: Temporal Difference
        self.d_conv = nn.Conv1d(diff_dim, 128, kernel_size=3, padding=1)
        self.d_bn = nn.BatchNorm1d(128)
        self.d_gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        
        # Fusion Multi-Head Attention
        self.fusion_dim = hidden_dim * 4 # 256
        self.attn = nn.MultiheadAttention(embed_dim=self.fusion_dim, num_heads=4, batch_first=True, dropout=0.20)
        self.norm = nn.LayerNorm(self.fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B, 8, 1024] -> split into spatial [B, 8, 512] and diff [B, 8, 512]
        x_s = x[:, :, :512]
        x_d = x[:, :, 512:]
        
        s_c = F.gelu(self.s_bn(self.s_conv(x_s.transpose(1, 2)))).transpose(1, 2)
        s_out, _ = self.s_gru(s_c)
        
        d_c = F.gelu(self.d_bn(self.d_conv(x_d.transpose(1, 2)))).transpose(1, 2)
        d_out, _ = self.d_gru(d_c)
        
        fused = torch.cat([s_out, d_out], dim=2) # [B, 8, 256]
        attn_out, _ = self.attn(fused, fused, fused)
        norm_out = self.norm(fused + attn_out)
        
        mean_p = norm_out.mean(dim=1)
        max_p = norm_out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# Pure Temporal Transformer (Candidate V6 Winner for reference/comparison)
class PureTemporalTransformer(nn.Module):
    def __init__(self, input_dim=512, d_model=256, nhead=4, num_layers=2, dropout=0.20):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
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
        h = self.projection(x) + self.pos_embedding
        out = self.transformer(h)
        out = self.norm(out)
        mean_p = out.mean(dim=1)
        max_p = out.max(dim=1).values
        pooled = torch.cat([mean_p, max_p], dim=1)
        return self.classifier(pooled).squeeze(1)

# ----------------------------------------------------------------------
# TRAINING & EVALUATION HARNESS
# ----------------------------------------------------------------------

def train_and_eval_model(
    model,
    X_tr, y_tr,
    X_v, y_v,
    X_te, y_te,
    exp_name="Experiment",
    epochs=15,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-3,
    pos_weight=7.438,
    mask_prob=0.0
):
    model = model.to(DEVICE)
    train_ds = RobustVideoDataset(X_tr, y_tr, mask_prob=mask_prob)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_tensor = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    test_tensor = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_score = -1.0
    best_state = None
    best_epoch = -1
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation Evaluation
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
            
    # Load best model from validation
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        train_logits = model(torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        val_logits = model(val_tensor).cpu().numpy()
        test_logits = model(test_tensor).cpu().numpy()
        
    # Fit Platt Calibrator strictly on validation logits
    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(val_logits.reshape(-1, 1), y_v)
    
    val_cal_probs = calibrator.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    test_cal_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    
    # Validation Threshold Search strictly on Validation Set
    best_t = 0.5
    best_t_f1_macro = -1.0
    for t in np.arange(0.10, 0.90, 0.01):
        pred = (val_cal_probs >= t).astype(int)
        f1_r = f1_score(y_v, pred, pos_label=1, zero_division=0)
        f1_f = f1_score(y_v, pred, pos_label=0, zero_division=0)
        bal_acc = balanced_accuracy_score(y_v, pred)
        rec_f = recall_score(y_v, pred, pos_label=0, zero_division=0)
        if rec_f >= 0.85: # protect fake detection integrity
            score = (f1_r + f1_f) / 2.0 + bal_acc * 0.5
            if score > best_t_f1_macro:
                best_t_f1_macro = score
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
    cm = confusion_matrix(y_te, test_pred) # TN (Fake as Fake), FP (Fake as Real), FN (Real as Fake), TP (Real as Real)
    tn, fp, fn, tp = cm.ravel()
    total_errors = fp + fn
    
    res = {
        "experiment": exp_name,
        "model_class": model.__class__.__name__,
        "pos_weight": pos_weight,
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
# EXECUTE CONTROLLED V7 EXPERIMENT SUITE
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("EXECUTING SYSTEMATIC CONTROLLED EXPERIMENTS (V4 Control vs. V7 Candidates)")
print("=" * 80)

results = []

# 1. Candidate A: V4 Control
print("\n--- Candidate A: V4 Control (Conv1D + BiGRU + MHA) ---")
res_a = train_and_eval_model(
    TemporalConvBiGRUControl(input_dim=512, hidden_dim=128),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V7_CandidateA_V4_Control",
    epochs=12, pos_weight=7.438
)
results.append(res_a)

# 2. Candidate B: V4 + Temporal Differences
print("\n--- Candidate B: V4 + Temporal Differences (1024D) ---")
res_b = train_and_eval_model(
    TemporalDiffBiGRUModel(input_dim=1024, hidden_dim=128),
    X_train_diff, y_train, X_val_diff, y_val, X_test_diff, y_test,
    exp_name="V7_CandidateB_TemporalDiffBiGRU",
    epochs=12, pos_weight=7.438
)
results.append(res_b)

# 3. Candidate C: CNN + Transformer
print("\n--- Candidate C: CNN + Transformer ---")
res_c = train_and_eval_model(
    CNNTemporalTransformer(input_dim=512, d_model=256, nhead=4, num_layers=2),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V7_CandidateC_CNNTemporalTransformer",
    epochs=12, pos_weight=7.438
)
results.append(res_c)

# 4. Candidate D: BiGRU + Transformer
print("\n--- Candidate D: BiGRU + Transformer ---")
res_d = train_and_eval_model(
    BiGRUTransformerModel(input_dim=512, hidden_dim=128, nhead=4, num_layers=2),
    X_train, y_train, X_val, y_val, X_test, y_test,
    exp_name="V7_CandidateD_BiGRUTransformer",
    epochs=12, pos_weight=7.438
)
results.append(res_d)

# 5. Candidate E1: Hybrid V7 (Conv + BiGRU + MHA + Attn Pooling + Diff)
print("\n--- Candidate E1: Hybrid V7 (lr=3e-4, wd=1e-3) ---")
res_e1 = train_and_eval_model(
    HybridV7Model(input_dim=1024, hidden_dim=128),
    X_train_diff, y_train, X_val_diff, y_val, X_test_diff, y_test,
    exp_name="V7_CandidateE1_HybridV7_Standard",
    epochs=15, lr=3e-4, weight_decay=1e-3, pos_weight=7.438
)
results.append(res_e1)

# 6. Candidate E2: Hybrid V7 with Moderate Class Weight (pos_w=5.5)
print("\n--- Candidate E2: Hybrid V7 (pos_weight=5.5) ---")
res_e2 = train_and_eval_model(
    HybridV7Model(input_dim=1024, hidden_dim=128),
    X_train_diff, y_train, X_val_diff, y_val, X_test_diff, y_test,
    exp_name="V7_CandidateE2_HybridV7_ModerateWeight",
    epochs=15, lr=3e-4, weight_decay=1e-3, pos_weight=5.50
)
results.append(res_e2)

# 7. Candidate F: Dual-Stream Temporal Model (Spatial + Diff)
print("\n--- Candidate F: Dual-Stream Temporal Model ---")
res_f = train_and_eval_model(
    DualStreamTemporalModel(spatial_dim=512, diff_dim=512, hidden_dim=64),
    X_train_diff, y_train, X_val_diff, y_val, X_test_diff, y_test,
    exp_name="V7_CandidateF_DualStreamModel",
    epochs=15, lr=3e-4, weight_decay=1e-3, pos_weight=7.438
)
results.append(res_f)

# 8. Candidate G: Robust Hybrid V7 with Missing-Face Frame Masking (p=0.15)
print("\n--- Candidate G: Hybrid V7 with Missing-Face Frame Masking (p=0.15) ---")
res_g = train_and_eval_model(
    HybridV7Model(input_dim=1024, hidden_dim=128),
    X_train_diff, y_train, X_val_diff, y_val, X_test_diff, y_test,
    exp_name="V7_CandidateG_RobustMaskedHybridV7",
    epochs=15, lr=3e-4, weight_decay=1e-3, pos_weight=7.438, mask_prob=0.15
)
results.append(res_g)

# ----------------------------------------------------------------------
# STEP 11: PROBABILITY ENSEMBLE FUSION EVALUATION
# ----------------------------------------------------------------------
print("\nSTEP 11: EVALUATING PROBABILITY ENSEMBLE FUSION (V4 Control + Top Candidates)...")
# Best single candidate based strictly on validation ROC-AUC / BalAcc
val_scores = [r["val_auc"] * 0.6 + r["val_bal_acc"] * 0.4 for r in results]
best_idx = int(np.argmax(val_scores))
best_cand = results[best_idx]
print(f"  Best Individual Candidate on Validation: {best_cand['experiment']} (Score: {val_scores[best_idx]:.4f})")

# Test validation fusion
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

# Threshold for fusion
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
    "experiment": f"V7_Fusion_V4_plus_{best_cand['experiment']}_(alpha={best_alpha:.1f})",
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
csv_out = os.path.join(REPORTS_DIR, "video_v7_experiments_summary.csv")
summary_df.to_csv(csv_out, index=False)
print(f"\n[SAVED] Experiments summary saved to: {csv_out}")

# Print Table Summary
print("\n" + "=" * 100)
print(f"{'Experiment':<35} | {'Val AUC':<8} | {'Val BalAcc':<10} | {'Test Acc':<9} | {'Test BalAcc':<11} | {'Test AUC':<9} | {'Real F1':<8} | {'Errors':<6}")
print("-" * 100)
for r in results:
    print(f"{r['experiment']:<35} | {r['val_auc']*100:>6.2f}% | {r['val_bal_acc']*100:>8.2f}% | "
          f"{r['test_acc']*100:>7.2f}% | {r['test_bal_acc']*100:>9.2f}% | {r['test_auc']*100:>7.2f}% | "
          f"{r['test_f1_real']*100:>6.2f}% | {r['total_errors']:>6}")
print("=" * 100)

# ----------------------------------------------------------------------
# STEP 16: STRICT SELECTION RULE VERIFICATION
# ----------------------------------------------------------------------
print("\nSTEP 16: STRICT PRODUCTION WINNER SELECTION...")
v4_errors = res_a["total_errors"]
v4_bal_acc = res_a["test_bal_acc"]
v4_auc = res_a["test_auc"]

print(f"  V4 Control Baseline: Test Acc={res_a['test_acc']*100:.2f}%, BalAcc={v4_bal_acc*100:.2f}%, "
      f"ROC-AUC={v4_auc*100:.2f}%, Real F1={res_a['test_f1_real']*100:.2f}%, Errors={v4_errors}")

# Find best V7 single architecture (excluding control)
v7_candidates = [r for r in results if "Control" not in r["experiment"] and "Fusion" not in r["experiment"]]
best_v7 = max(v7_candidates, key=lambda x: (x["val_auc"] * 0.6 + x["val_bal_acc"] * 0.4))

print(f"\n  Top V7 Standalone Candidate: {best_v7['experiment']}")
print(f"    Validation: AUC={best_v7['val_auc']*100:.2f}%, BalAcc={best_v7['val_bal_acc']*100:.2f}%, Real F1={best_v7['val_f1_real']*100:.2f}%")
print(f"    Test      : Acc={best_v7['test_acc']*100:.2f}%, BalAcc={best_v7['test_bal_acc']*100:.2f}%, ROC-AUC={best_v7['test_auc']*100:.2f}%, "
      f"Real F1={best_v7['test_f1_real']*100:.2f}%, Errors={best_v7['total_errors']}")

# Check if V7 genuinely improves over V4
v7_is_winner = (
    best_v7["test_bal_acc"] > v4_bal_acc and
    best_v7["test_f1_real"] > res_a["test_f1_real"] and
    best_v7["total_errors"] <= v4_errors
)

print(f"\n  Selection Criteria Status: Genuine Win = {v7_is_winner}")

# Save the V7 candidate checkpoint and artifacts
OUT_V7_DIR = os.path.join(BASE_DIR, "models", "video_final_v7")
os.makedirs(OUT_V7_DIR, exist_ok=True)

torch.save(best_v7["model_state"], os.path.join(OUT_V7_DIR, "temporal_attention_model.pth"))

with open(os.path.join(OUT_V7_DIR, "video_scaler.pkl"), "wb") as f:
    pickle.dump({"mean": feat_mean, "std": feat_std}, f)

with open(os.path.join(OUT_V7_DIR, "probability_calibrator.pkl"), "wb") as f:
    pickle.dump({"model": best_v7["calibrator"], "threshold": best_v7["threshold"]}, f)

with open(os.path.join(OUT_V7_DIR, "video_config.pkl"), "wb") as f:
    pickle.dump({
        "model_name": "VideoDetector_V7",
        "architecture": best_v7["model_class"],
        "experiment": best_v7["experiment"],
        "frames_sampled": 8,
        "face_feature_dim": 512,
        "input_dim": 1024 if "Diff" in best_v7["model_class"] or "Hybrid" in best_v7["model_class"] or "DualStream" in best_v7["model_class"] else 512,
        "decision_threshold": best_v7["threshold"]
    }, f)

with open(os.path.join(OUT_V7_DIR, "video_metrics.pkl"), "wb") as f:
    pickle.dump(best_v7, f)

print(f"[SAVED] VideoDetector_V7 model and artifacts saved to: {OUT_V7_DIR}")
print("\nV7 EXPERIMENT CAMPAIGN COMPLETE.")
