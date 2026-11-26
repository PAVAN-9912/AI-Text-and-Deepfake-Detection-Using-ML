"""
Phase 7F.4 — Controlled ST-SeqNet Exploratory Training Script

Strict Requirements Enforced:
1. Isolation:
   - Trains ONLY on the 400-video development training partition across all 8 categories (including all 50 C6 samples).
   - Never touches or loads the sealed 200-video independent test partition (guarded by PermissionError).
   - Never touches Phase 7E quarantined data.
   - Zero modifications to production FaceViT, thresholds (0.475), or shadow mode (0).
   - All checkpoints saved strictly under experiments/phase7f_4/checkpoints/.
2. Conservative Hardware Settings:
   - Device: CPU only.
   - Batch size: 2.
   - Workers: 0, pin_memory: False.
   - Max epochs: 10, early stopping patience: 3 epochs.
   - Gradient clipping: max_norm = 1.0.
   - System RAM monitoring (safety halt if available RAM < 500 MB).
3. Dual Validation Protocols:
   - Protocol A: Full validation (N=120, C1-C8).
   - Protocol B: Identity-disjoint validation (N=90, C1, C3, C4, C5, C7, C8).
   - Fixed predeclared classification threshold = 0.50.
"""

import os
import sys
import time
import random
import ctypes
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"C:\Users\PAVAN S\OneDrive\Desktop\AI_Content_Detection"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experimental.phase7f.preprocessing import StandardVideoPreprocessor
from src.experimental.phase7f.augmentation import AntiShortcutVideoAugmentor, AugmentationConfig
from src.experimental.phase7f.dataset import STSeqNetDataset, safe_collate_fn
from src.experimental.phase7f.models import STSeqNet

# Global configurations
SEED = 42
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE = 2
LR_BACKBONE = 2e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
PRIMARY_THRESHOLD = 0.50
MIN_AVAIL_RAM_MB = 500

EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase7f_4")
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase7_external_evaluation", "PHASE7F_2_PROVENANCE_MANIFEST.csv")
EPOCH_METRICS_CSV = os.path.join(EXPERIMENT_DIR, "phase7f_4_epoch_metrics.csv")
PER_VIDEO_PREDICTIONS_CSV = os.path.join(EXPERIMENT_DIR, "phase7f_4_per_video_predictions.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_available_ram_mb() -> float:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return float(stat.ullAvailPhys / (1024**2))


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # AI Recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Genuine Specificity
    balanced_acc = (sensitivity + specificity) / 2.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "sensitivity_ai_recall": sensitivity,
        "specificity_genuine": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "precision": precision,
        "f1_score": f1,
        "roc_auc": auc,
    }


def evaluate_dataset(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    model.eval()
    losses = []
    y_true_list = []
    y_prob_list = []
    records = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inputs, labels, metas = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            logits = outputs["logits"]
            probs = outputs["probabilities"].squeeze(-1).cpu().numpy()
            batch_loss = criterion(logits, labels).item()
            losses.append(batch_loss)

            batch_labels = labels.squeeze(-1).cpu().numpy().astype(int)
            y_true_list.extend(batch_labels)
            y_prob_list.extend(probs)

            for i in range(len(metas)):
                meta = metas[i]
                meta_pred = {
                    "sample_id": meta["sample_id"],
                    "category": meta["category"],
                    "ground_truth": int(batch_labels[i]),
                    "predicted_prob": float(probs[i]),
                    "predicted_class": int(probs[i] >= PRIMARY_THRESHOLD),
                    "identity_overlap_flag": meta["identity_overlap_flag"],
                }
                records.append(meta_pred)

    y_true_arr = np.array(y_true_list)
    y_prob_arr = np.array(y_prob_list)
    metrics = calculate_metrics(y_true_arr, y_prob_arr, threshold=PRIMARY_THRESHOLD)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["valid_samples"] = len(y_true_arr)

    return metrics, records


def run_training():
    set_seed(SEED)
    device = torch.device("cpu")
    print(f"[Training] Initiating Phase 7F.4 ST-SeqNet training on {device}...")
    print(f"[Training] Hyperparameters: BS={BATCH_SIZE}, LR_backbone={LR_BACKBONE}, LR_head={LR_HEAD}, WD={WEIGHT_DECAY}, Seed={SEED}")

    # Datasets
    augmentor = AntiShortcutVideoAugmentor(AugmentationConfig(enabled=True))
    train_ds = STSeqNetDataset(
        manifest_path=MANIFEST_PATH,
        partition="train",
        project_root=PROJECT_ROOT,
        augmentor=augmentor,
        filter_identity_overlap=False,
    )
    val_full_ds = STSeqNetDataset(
        manifest_path=MANIFEST_PATH,
        partition="val",
        project_root=PROJECT_ROOT,
        augmentor=None,
        filter_identity_overlap=False,
    )
    val_disjoint_ds = STSeqNetDataset(
        manifest_path=MANIFEST_PATH,
        partition="val",
        project_root=PROJECT_ROOT,
        augmentor=None,
        filter_identity_overlap=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=safe_collate_fn, num_workers=0)
    val_full_loader = DataLoader(val_full_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate_fn, num_workers=0)
    val_disjoint_loader = DataLoader(val_disjoint_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate_fn, num_workers=0)

    print(f"[Training] Loaded datasets: Train={len(train_ds)}, Val_Full={len(val_full_ds)}, Val_Disjoint={len(val_disjoint_ds)}")

    # Model
    model = STSeqNet(pretrained_backbone=True, freeze_backbone=False).to(device)

    # Differential learning rates
    backbone_params = list(model.features.parameters())
    other_params = (
        list(model.feature_diff_projection.parameters())
        + list(model.gru.parameters())
        + list(model.attention.parameters())
        + list(model.classifier.parameters())
    )
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": other_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    criterion = nn.BCEWithLogitsLoss()

    best_val_score = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    epoch_logs = []
    best_records = []

    total_start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        valid_train_count = 0

        # RAM check before epoch
        avail_ram = get_available_ram_mb()
        print(f"\n--- Epoch {epoch}/{MAX_EPOCHS} | Avail RAM: {avail_ram:.1f} MB ---", flush=True)
        if avail_ram < MIN_AVAIL_RAM_MB:
            print(f"[SAFETY HALT] Low available RAM: {avail_ram:.1f} MB < {MIN_AVAIL_RAM_MB} MB. Stopping.", flush=True)
            break

        step_count = 0
        for batch in train_loader:
            if batch is None:
                continue
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

            # Check finite loss
            if not torch.isfinite(loss):
                print(f"[SAFETY HALT] Non-finite loss detected at epoch {epoch}: {loss.item()}. Stopping.", flush=True)
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            train_losses.append(loss.item())
            valid_train_count += inputs.shape[0]
            step_count += 1
            if step_count % 50 == 0:
                print(f"  [Epoch {epoch}] Step {step_count}/{len(train_loader)} | Running Loss: {np.mean(train_losses[-50:]):.4f} | RAM: {get_available_ram_mb():.1f} MB", flush=True)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Evaluate both validation protocols
        val_full_metrics, val_full_records = evaluate_dataset(model, val_full_loader, criterion, device)
        val_disjoint_metrics, _ = evaluate_dataset(model, val_disjoint_loader, criterion, device)

        epoch_duration = time.time() - epoch_start
        print(f"[Epoch {epoch} Complete] Time: {epoch_duration:.1f}s | Train Loss: {mean_train_loss:.4f}", flush=True)
        print(f"  Protocol A (Full Val, N={val_full_metrics['valid_samples']}): Loss={val_full_metrics['loss']:.4f}, Acc={val_full_metrics['accuracy']:.4f}, BalAcc={val_full_metrics['balanced_accuracy']:.4f}, F1={val_full_metrics['f1_score']:.4f}, AUC={val_full_metrics['roc_auc']:.4f}", flush=True)
        print(f"  Protocol B (Disjoint Val, N={val_disjoint_metrics['valid_samples']}): Loss={val_disjoint_metrics['loss']:.4f}, Acc={val_disjoint_metrics['accuracy']:.4f}, BalAcc={val_disjoint_metrics['balanced_accuracy']:.4f}, F1={val_disjoint_metrics['f1_score']:.4f}, AUC={val_disjoint_metrics['roc_auc']:.4f}", flush=True)

        # Record epoch metrics
        log_entry = {
            "epoch": epoch,
            "epoch_duration_sec": round(epoch_duration, 2),
            "train_loss": round(mean_train_loss, 4),
            "train_valid_samples": valid_train_count,
            "avail_ram_mb": round(get_available_ram_mb(), 1),
            # Protocol A
            "protoA_loss": round(val_full_metrics["loss"], 4),
            "protoA_accuracy": round(val_full_metrics["accuracy"], 4),
            "protoA_balanced_acc": round(val_full_metrics["balanced_accuracy"], 4),
            "protoA_ai_recall": round(val_full_metrics["sensitivity_ai_recall"], 4),
            "protoA_genuine_spec": round(val_full_metrics["specificity_genuine"], 4),
            "protoA_fpr": round(val_full_metrics["false_positive_rate"], 4),
            "protoA_f1": round(val_full_metrics["f1_score"], 4),
            "protoA_auc": round(val_full_metrics["roc_auc"], 4),
            # Protocol B
            "protoB_loss": round(val_disjoint_metrics["loss"], 4),
            "protoB_accuracy": round(val_disjoint_metrics["accuracy"], 4),
            "protoB_balanced_acc": round(val_disjoint_metrics["balanced_accuracy"], 4),
            "protoB_ai_recall": round(val_disjoint_metrics["sensitivity_ai_recall"], 4),
            "protoB_genuine_spec": round(val_disjoint_metrics["specificity_genuine"], 4),
            "protoB_fpr": round(val_disjoint_metrics["false_positive_rate"], 4),
            "protoB_f1": round(val_disjoint_metrics["f1_score"], 4),
            "protoB_auc": round(val_disjoint_metrics["roc_auc"], 4),
        }
        epoch_logs.append(log_entry)

        # Save epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"st_seqnet_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "protoA_metrics": val_full_metrics,
            "protoB_metrics": val_disjoint_metrics,
        }, ckpt_path)
        log_entry["checkpoint_path"] = ckpt_path
        log_entry["checkpoint_sha256"] = compute_sha256(ckpt_path)

        # Checkpoint selection: Monitor Protocol A Balanced Accuracy
        current_score = val_full_metrics["balanced_accuracy"]
        if current_score > best_val_score:
            best_val_score = current_score
            best_epoch = epoch
            no_improve_epochs = 0
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "st_seqnet_best_checkpoint.pth")
            torch.save({
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "protoA_metrics": val_full_metrics,
                "protoB_metrics": val_disjoint_metrics,
            }, best_ckpt_path)
            best_records = val_full_records
            print(f"  [*] New best model saved at epoch {epoch} (ProtoA BalAcc: {best_val_score:.4f})", flush=True)
        else:
            no_improve_epochs += 1
            print(f"  [!] No improvement in ProtoA BalAcc for {no_improve_epochs} epoch(s).", flush=True)
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"[EARLY STOPPING] Patience of {EARLY_STOPPING_PATIENCE} epochs reached. Terminating training.", flush=True)
                break

    total_training_duration = time.time() - total_start_time
    print(f"\n[Training Complete] Total Duration: {total_training_duration:.1f}s ({total_training_duration/60:.2f} min). Best Epoch: {best_epoch}", flush=True)

    # Save metrics CSV
    metrics_df = pd.DataFrame(epoch_logs)
    metrics_df.to_csv(EPOCH_METRICS_CSV, index=False)
    print(f"[Artifacts] Saved epoch metrics: {EPOCH_METRICS_CSV}", flush=True)

    # Save best predictions CSV
    if best_records:
        preds_df = pd.DataFrame(best_records)
        preds_df.to_csv(PER_VIDEO_PREDICTIONS_CSV, index=False)
        print(f"[Artifacts] Saved per-video predictions: {PER_VIDEO_PREDICTIONS_CSV}", flush=True)


if __name__ == "__main__":
    run_training()
