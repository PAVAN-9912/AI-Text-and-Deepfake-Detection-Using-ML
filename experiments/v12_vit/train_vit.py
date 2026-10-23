"""
Vision Transformer (ViT) Deepfake Video Detector Training
=========================================================
Trains a fine-tuned Vision Transformer (google/vit-base-patch16-224)
on face crops from Celeb-DF v2 dataset.

Architecture:
  - Backbone: google/vit-base-patch16-224 (pre-trained on ImageNet-21k, fine-tuned on 1k)
  - Fine-tuning: Top Transformer Encoder Block (Layer 11) + Layernorm + Classification Head
  - Class Mapping: 0 = FAKE, 1 = REAL
  - Normalization: Mean [0.5, 0.5, 0.5], Std [0.5, 0.5, 0.5] (ViT native standard)
"""

import sys
import os
import io
import time
import json
import random
import hashlib
from pathlib import Path
from typing import List, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from transformers import ViTForImageClassification
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "experiments" / "v12_vit"
CROPS_DIR = BASE_DIR / "experiments" / "v11_multigenerator" / "data" / "crops"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
SAMPLES_PER_CLASS_TRAIN = 2500  # 2,500 real + 2,500 fake = 5,000 per epoch
SAMPLES_PER_CLASS_VAL = 800     # 800 real + 800 fake = 1,600 per validation
LR = 1e-4
WEIGHT_DECAY = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=VIT_MEAN, std=VIT_STD),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


class BalancedFaceCropDataset(Dataset):
    """Loads face crops with class balance (50% Real, 50% Fake)."""

    def __init__(self, split: str, transform=None, samples_per_class: int = 2500, shuffle_subset: bool = True):
        self.transform = transform
        self.split_dir = CROPS_DIR / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Crops directory not found: {self.split_dir}")

        real_crops = sorted([p for p in (self.split_dir / "real").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "_nf" not in p.stem])
        fake_crops = sorted([p for p in (self.split_dir / "fake").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "_nf" not in p.stem])

        if shuffle_subset:
            random.shuffle(real_crops)
            random.shuffle(fake_crops)

        num_real = min(samples_per_class, len(real_crops))
        num_fake = min(samples_per_class, len(fake_crops))

        selected_real = real_crops[:num_real]
        selected_fake = fake_crops[:num_fake]

        self.samples: List[Tuple[Path, int]] = (
            [(p, 1) for p in selected_real] + [(p, 0) for p in selected_fake]
        )
        random.shuffle(self.samples)

        print(f"[{split.upper()}] Dataset ready: {len(self.samples)} samples (Real={num_real}, Fake={num_fake})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_vit_model():
    print("Loading pretrained google/vit-base-patch16-224...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
        ignore_mismatched_sizes=True,
    )

    # Freeze base transformer layers 0 to 10 and patch embeddings
    for name, param in model.named_parameters():
        if any(k in name for k in ["layers.11", "vit.layernorm", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"ViT Model Architecture: {trainable:,} trainable params out of {total:,} ({trainable/total*100:.2f}%)")
    return model


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_probs_real = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs.logits, targets)
            val_loss += loss.item() * len(targets)

            probs = torch.softmax(outputs.logits, dim=-1)
            probs_real = probs[:, 1].cpu().numpy()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()

            all_targets.extend(targets.cpu().numpy())
            all_probs_real.extend(probs_real)
            all_preds.extend(preds)

    val_loss /= len(val_loader.dataset)
    auc = roc_auc_score(all_targets, all_probs_real)
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)

    return val_loss, acc, bal_acc, auc


def train():
    print("=" * 70)
    print("STARTING VISION TRANSFORMER (ViT) TRAINING FOR DEEPFAKE DETECTION")
    print("=" * 70)

    val_dataset = BalancedFaceCropDataset(
        "validation",
        transform=val_transform,
        samples_per_class=SAMPLES_PER_CLASS_VAL,
        shuffle_subset=False
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_vit_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_auc = 0.0
    best_ckpt_path = CKPT_DIR / "vit_deepfake_best.pth"
    log_file = RESULTS_DIR / "vit_training_log.csv"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_acc,val_bal_acc,val_auc,time_sec\n")

    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        # Fresh balanced subset per epoch
        train_dataset = BalancedFaceCropDataset(
            "train",
            transform=train_transform,
            samples_per_class=SAMPLES_PER_CLASS_TRAIN,
            shuffle_subset=True
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        print(f"\n--- Epoch {epoch}/{EPOCHS} (Learning Rate: {scheduler.get_last_lr()[0]:.2e}) ---")
        for step, (images, targets) in enumerate(train_loader, 1):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)

            if step % 25 == 0 or step == total_batches:
                avg_step_loss = running_loss / (step * BATCH_SIZE)
                elapsed = time.time() - epoch_start
                print(f"  Batch [{step}/{total_batches}] | Loss: {avg_step_loss:.4f} | Elapsed: {elapsed:.1f}s")

        epoch_train_loss = running_loss / len(train_dataset)
        val_loss, val_acc, val_bal_acc, val_auc = evaluate(model, val_loader, criterion)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch} Complete | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Bal Acc: {val_bal_acc:.4f} | Val AUC: {val_auc:.4f} | Time: {epoch_time:.1f}s")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_train_loss:.4f},{val_loss:.4f},{val_acc:.4f},{val_bal_acc:.4f},{val_auc:.4f},{epoch_time:.1f}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            print(f"  >>> New Best Model Found (Val AUC: {val_auc:.4f})! Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "val_bal_acc": val_bal_acc,
                "val_loss": val_loss,
                "train_loss": epoch_train_loss,
                "class_mapping": {0: "FAKE", 1: "REAL"},
                "backbone": "google/vit-base-patch16-224",
                "normalization": {"mean": VIT_MEAN, "std": VIT_STD},
                "input_size": 224,
            }, best_ckpt_path)

            sha256_hash = hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest()
            meta = {
                "checkpoint": str(best_ckpt_path.name),
                "sha256": sha256_hash,
                "epoch": epoch,
                "val_auc": float(val_auc),
                "val_accuracy": float(val_acc),
                "val_balanced_accuracy": float(val_bal_acc),
                "backbone": "google/vit-base-patch16-224",
                "device": str(DEVICE),
                "train_samples_per_epoch": len(train_dataset),
                "val_samples": len(val_dataset),
            }
            with open(CKPT_DIR / "vit_best_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"  >>> Checkpoint saved: {best_ckpt_path} (SHA-256: {sha256_hash[:16]}...)")

    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Best Checkpoint: {best_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()
