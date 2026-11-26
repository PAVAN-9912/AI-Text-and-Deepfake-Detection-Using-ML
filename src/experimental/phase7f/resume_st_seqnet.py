"""
Phase 7F.4 — Resume ST-SeqNet Training Script (Batch-Controlled)

Supports resuming training in exact 2-epoch increments (e.g. epochs 5-6, 7-8, 9-10).
Restores complete training state:
- Model weights
- Optimizer state (momentum & variance buffers)
- RNG states (PyTorch, NumPy, Python)
- Training history and best validation tracking
"""

import os
import sys
import time
import random
import ctypes
import hashlib
import argparse
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

from src.experimental.phase7f.augmentation import AntiShortcutVideoAugmentor, AugmentationConfig
from src.experimental.phase7f.dataset import STSeqNetDataset, safe_collate_fn
from src.experimental.phase7f.models import STSeqNet

EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase7f_4")
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase7_external_evaluation", "PHASE7F_2_PROVENANCE_MANIFEST.csv")
METRICS_CSV = os.path.join(EXPERIMENT_DIR, "phase7f_4_epoch_metrics.csv")
PRIMARY_THRESHOLD = 0.50
MAX_GRAD_NORM = 1.0
MIN_AVAIL_RAM_MB = 500


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
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
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
                    "is_correct": int((probs[i] >= PRIMARY_THRESHOLD) == batch_labels[i]),
                    "identity_overlap_flag": meta["identity_overlap_flag"],
                }
                records.append(meta_pred)

    y_true_arr = np.array(y_true_list)
    y_prob_arr = np.array(y_prob_list)
    metrics = calculate_metrics(y_true_arr, y_prob_arr, threshold=PRIMARY_THRESHOLD)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["valid_samples"] = len(y_true_arr)

    return metrics, records


def resume_training(checkpoint_path: str, epochs_to_run: int = 2):
    device = torch.device("cpu")
    print(f"[Resume] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    start_epoch = state["next_epoch"]
    end_epoch = start_epoch + epochs_to_run - 1
    max_epochs = state.get("max_epochs", 10)
    end_epoch = min(end_epoch, max_epochs)

    print(f"[Resume] Current completed epoch: {state['epoch']} -> Running epochs {start_epoch} to {end_epoch}")

    # Restore RNG states
    if "torch_rng_state" in state and state["torch_rng_state"] is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if "numpy_rng_state" in state and state["numpy_rng_state"] is not None:
        np.random.set_state(state["numpy_rng_state"])
    if "python_rng_state" in state and state["python_rng_state"] is not None:
        random.setstate(state["python_rng_state"])

    # Load Model
    model = STSeqNet(pretrained_backbone=False).to(device)
    model.load_state_dict(state["model_state_dict"])

    # Optimizer
    lr_backbone = state.get("learning_rate_backbone", 2e-5)
    lr_head = state.get("learning_rate_head", 1e-4)
    weight_decay = state.get("weight_decay", 1e-4)

    backbone_params = list(model.features.parameters())
    other_params = (
        list(model.feature_diff_projection.parameters())
        + list(model.gru.parameters())
        + list(model.attention.parameters())
        + list(model.classifier.parameters())
    )
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr_head, "weight_decay": weight_decay},
    ])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    criterion = nn.BCEWithLogitsLoss()

    best_val_score = state.get("best_val_score", 0.0)
    best_epoch = state.get("best_epoch", 1)
    history = state.get("training_history", [])

    # Datasets
    augmentor = AntiShortcutVideoAugmentor(AugmentationConfig(enabled=True))
    train_ds = STSeqNetDataset(manifest_path=MANIFEST_PATH, partition="train", project_root=PROJECT_ROOT, augmentor=augmentor, filter_identity_overlap=False)
    val_full_ds = STSeqNetDataset(manifest_path=MANIFEST_PATH, partition="val", project_root=PROJECT_ROOT, augmentor=None, filter_identity_overlap=False)
    val_disjoint_ds = STSeqNetDataset(manifest_path=MANIFEST_PATH, partition="val", project_root=PROJECT_ROOT, augmentor=None, filter_identity_overlap=True)

    batch_size = state.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=safe_collate_fn, num_workers=0)
    val_full_loader = DataLoader(val_full_ds, batch_size=batch_size, shuffle=False, collate_fn=safe_collate_fn, num_workers=0)
    val_disjoint_loader = DataLoader(val_disjoint_ds, batch_size=batch_size, shuffle=False, collate_fn=safe_collate_fn, num_workers=0)

    val_full_metrics = None
    val_disjoint_metrics = None
    last_completed_epoch = None
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        valid_train_count = 0

        avail_ram = get_available_ram_mb()
        print(f"\n--- Epoch {epoch}/{max_epochs} | Avail RAM: {avail_ram:.1f} MB ---", flush=True)
        if avail_ram < MIN_AVAIL_RAM_MB:
            print(f"[SAFETY HALT] Low available RAM: {avail_ram:.1f} MB < {MIN_AVAIL_RAM_MB} MB. Halting.", flush=True)
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

            if not torch.isfinite(loss):
                print(f"[SAFETY HALT] Non-finite loss at epoch {epoch}: {loss.item()}. Stopping.", flush=True)
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

        log_entry = {
            "epoch": epoch,
            "epoch_duration_sec": round(epoch_duration, 2),
            "train_loss": round(mean_train_loss, 4),
            "train_valid_samples": valid_train_count,
            "avail_ram_mb": round(get_available_ram_mb(), 1),
            "protoA_loss": round(val_full_metrics["loss"], 4),
            "protoA_accuracy": round(val_full_metrics["accuracy"], 4),
            "protoA_balanced_acc": round(val_full_metrics["balanced_accuracy"], 4),
            "protoA_ai_recall": round(val_full_metrics["sensitivity_ai_recall"], 4),
            "protoA_genuine_spec": round(val_full_metrics["specificity_genuine"], 4),
            "protoA_fpr": round(val_full_metrics["false_positive_rate"], 4),
            "protoA_f1": round(val_full_metrics["f1_score"], 4),
            "protoA_auc": round(val_full_metrics["roc_auc"], 4),
            "protoB_loss": round(val_disjoint_metrics["loss"], 4),
            "protoB_accuracy": round(val_disjoint_metrics["accuracy"], 4),
            "protoB_balanced_acc": round(val_disjoint_metrics["balanced_accuracy"], 4),
            "protoB_ai_recall": round(val_disjoint_metrics["sensitivity_ai_recall"], 4),
            "protoB_genuine_spec": round(val_disjoint_metrics["specificity_genuine"], 4),
            "protoB_fpr": round(val_disjoint_metrics["false_positive_rate"], 4),
            "protoB_f1": round(val_disjoint_metrics["f1_score"], 4),
            "protoB_auc": round(val_disjoint_metrics["roc_auc"], 4),
        }
        history.append(log_entry)

        # Save standard epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"st_seqnet_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "protoA_metrics": val_full_metrics,
            "protoB_metrics": val_disjoint_metrics,
        }, ckpt_path)

        # Always save per-video predictions for this epoch
        pred_csv_path = os.path.join(EXPERIMENT_DIR, f"phase7f_4_per_video_predictions_epoch{epoch}.csv")
        df_records = pd.DataFrame(val_full_records)
        df_records.to_csv(pred_csv_path, index=False)
        
        # Print category breakdown
        print(f"  Category Breakdown (Epoch {epoch}):", flush=True)
        cat_summary = df_records.groupby("category")["is_correct"].agg(["count", "sum", "mean"])
        for cat_name, row in cat_summary.iterrows():
            print(f"    {cat_name:20s}: {int(row['sum'])}/{int(row['count'])} ({row['mean']*100:.1f}%)", flush=True)

        # Update best checkpoint if improved
        if val_full_metrics["balanced_accuracy"] > best_val_score:
            best_val_score = val_full_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "st_seqnet_best_checkpoint.pth")
            torch.save({
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "protoA_metrics": val_full_metrics,
                "protoB_metrics": val_disjoint_metrics,
            }, best_ckpt_path)
            print(f"  [*] New best model saved at epoch {epoch} (ProtoA BalAcc: {best_val_score:.4f})", flush=True)
        else:
            print(f"  [-] Best model retained at epoch {best_epoch} (ProtoA BalAcc: {best_val_score:.4f})", flush=True)

        last_completed_epoch = epoch

    if last_completed_epoch is None:
        print("[Resume Error] No epochs were completed.", flush=True)
        return

    # Save next resumable checkpoint
    next_resumable_path = os.path.join(CHECKPOINT_DIR, f"st_seqnet_resumable_epoch_{last_completed_epoch}.pth")
    resumable_payload = {
        "epoch": last_completed_epoch,
        "next_epoch": last_completed_epoch + 1,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate_backbone": lr_backbone,
        "learning_rate_head": lr_head,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "training_history": history,
        "protoA_metrics_last": val_full_metrics,
        "protoB_metrics_last": val_disjoint_metrics,
    }
    torch.save(resumable_payload, next_resumable_path)
    print(f"\n[Resume Complete] Saved complete resumable checkpoint for next session: {next_resumable_path} (Next epoch: {end_epoch + 1})")

    # Update metrics CSV
    pd.DataFrame(history).to_csv(METRICS_CSV, index=False)
    print(f"[Artifacts] Updated epoch metrics: {METRICS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-checkpoint", type=str, default=os.path.join(CHECKPOINT_DIR, "st_seqnet_resumable_epoch_4.pth"))
    parser.add_argument("--epochs-to-run", type=int, default=2)
    args = parser.parse_args()

    resume_training(checkpoint_path=args.resume_checkpoint, epochs_to_run=args.epochs_to_run)
