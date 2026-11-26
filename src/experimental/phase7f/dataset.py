"""
PyTorch Dataset and Metadata Isolation Safeguards for Phase 7F.3

Key Design Requirements:
1. Metadata Isolation: Only the preprocessed (16, 3, 224, 224) tensor and binary label (0.0=real, 1.0=ai)
   are supplied to the model. Filepaths, codecs, sources, container metadata, and category labels are stripped.
2. Identity Safeguards: Preserves donor, target, and genuine identities in sidecar metadata dicts.
   Explicitly flags shared identity samples:
   - id0 (C2 genuine vs C6 manipulated in Train)
   - id40 (C2 genuine vs C6 manipulated in Val)
   - id41 (C2 genuine vs C6 manipulated in Val)
   Enables evaluating on complete validation vs identity-disjoint validation subsets.
3. Independent Test Set Lock: Hard guard preventing the 200-video test partition from being loaded for training
   or hyperparameter tuning.
"""

import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Tuple

from .preprocessing import StandardVideoPreprocessor, PreprocessingConfig
from .augmentation import AntiShortcutVideoAugmentor, AugmentationConfig


class STSeqNetDataset(Dataset):
    """
    Standardized Dataset for ST-SeqNet training, validation, and evaluation.
    """

    def __init__(
        self,
        manifest_path: str,
        partition: str = "train",
        project_root: Optional[str] = None,
        preprocessor: Optional[StandardVideoPreprocessor] = None,
        augmentor: Optional[AntiShortcutVideoAugmentor] = None,
        allow_test_split: bool = False,
        filter_identity_overlap: bool = False,
    ):
        """
        Args:
            manifest_path: Path to PHASE7F_2_PROVENANCE_MANIFEST.csv
            partition: 'train', 'val', or 'test'
            project_root: Root directory of project to resolve relative filepaths
            preprocessor: StandardVideoPreprocessor instance
            augmentor: AntiShortcutVideoAugmentor instance (active for train by default)
            allow_test_split: Explicit security gate to access test partition (strictly locked)
            filter_identity_overlap: If True, filters out shared identity samples (id0 in train; id40, id41 in val)
        """
        assert partition in ["train", "val", "test"], f"Invalid partition: {partition}"
        if partition == "test" and not allow_test_split:
            raise PermissionError(
                "ACCESS DENIED: Independent Test cohort is strictly locked to prevent leakage during Phase 7F.3."
            )

        self.partition = partition
        self.project_root = project_root or os.getcwd()
        self.preprocessor = preprocessor or StandardVideoPreprocessor()
        self.augmentor = augmentor
        self.failed_decodings: List[Dict[str, Any]] = []

        # Load manifest
        df = pd.read_csv(manifest_path)
        self.df = df[df["partition"] == partition].copy().reset_index(drop=True)

        # Parse and flag identity relationships
        self._audit_and_flag_identities()

        if filter_identity_overlap:
            initial_count = len(self.df)
            self.df = self.df[~self.df["identity_overlap_flag"]].copy().reset_index(drop=True)
            print(
                f"[Dataset] Applied identity filter on '{partition}': retained {len(self.df)}/{initial_count} samples."
            )

    def _audit_and_flag_identities(self):
        """
        Extracts donor, target, and genuine identities and flags cross-authenticity shared identities.
        """
        donor_ids = []
        target_ids = []
        genuine_ids = []
        overlap_flags = []

        for idx, row in self.df.iterrows():
            cat = row["category"]
            ident = str(row["subject_identity"])
            d_id = None
            t_id = None
            g_id = None
            is_overlap = False

            if cat == "C2_real_face_man":
                m = re.match(r"celeb_id_(\d+)", ident)
                if m:
                    g_id = int(m.group(1))
                    # Flag id0 in train, id40 or id41 in val
                    if (self.partition == "train" and g_id == 0) or (
                        self.partition == "val" and g_id in [40, 41]
                    ):
                        is_overlap = True
            elif cat == "C6_ai_face_man":
                m = re.match(r"donor_(\d+)_target_(\d+)", ident)
                if m:
                    d_id = int(m.group(1))
                    t_id = int(m.group(2))
                    # Flag donor 0 in train, donor 40 or target 41 in val
                    if (self.partition == "train" and d_id == 0) or (
                        self.partition == "val" and (d_id == 40 or t_id == 41)
                    ):
                        is_overlap = True

            donor_ids.append(d_id)
            target_ids.append(t_id)
            genuine_ids.append(g_id)
            overlap_flags.append(is_overlap)

        self.df["donor_identity"] = donor_ids
        self.df["target_identity"] = target_ids
        self.df["genuine_identity"] = genuine_ids
        self.df["identity_overlap_flag"] = overlap_flags

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            frames_tensor: torch.FloatTensor of shape (16, 3, 224, 224)
            label: torch.FloatTensor of shape (1,) (0.0 for genuine, 1.0 for ai_generated)
            sidecar_meta: Dict containing sample_id and identity flags for separate evaluation
        """
        row = self.df.iloc[idx]
        fp = row["filepath"]
        if not os.path.isabs(fp):
            fp = os.path.join(self.project_root, fp)

        # 1. Decode video to uint8 RGB frames (16, 224, 224, 3)
        frames_array, status = self.preprocessor.decode_video(fp)
        if not status["success"] or frames_array is None:
            # Exclude failed video: log failure in audit records and return None
            # Zero-tensor fallback samples are strictly prevented from entering model batches
            failure_record = {
                "sample_id": row["sample_id"],
                "filepath": fp,
                "partition": self.partition,
                "category": row["category"],
                "error_msg": status.get("error_msg", "Unknown decoding error"),
            }
            self.failed_decodings.append(failure_record)
            return None

        # 2. Apply anti-shortcut augmentation if enabled
        if self.augmentor is not None:
            frames_array = self.augmentor.augment_clip(frames_array)

        # 3. Convert to normalized float tensor: shape (16, 3, 224, 224)
        tensor = self.preprocessor.frames_to_tensor(frames_array)

        # 4. Binary label
        # 0.0 = genuine, 1.0 = ai_generated
        is_ai = 1.0 if row["ground_truth_class"] == "ai_generated" else 0.0
        label_tensor = torch.tensor([is_ai], dtype=torch.float32)

        # 5. Sidecar audit metadata (DO NOT feed to model)
        sidecar_meta = {
            "sample_id": row["sample_id"],
            "partition": self.partition,
            "category": row["category"],
            "identity_overlap_flag": bool(row["identity_overlap_flag"]),
            "genuine_identity": row["genuine_identity"],
            "donor_identity": row["donor_identity"],
            "target_identity": row["target_identity"],
            "decoding_success": True,
        }

        return tensor, label_tensor, sidecar_meta


def safe_collate_fn(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]]:
    """
    Safe PyTorch DataLoader collate_fn that strictly excludes failed/corrupted video entries.
    Zero-tensor fallback samples can never enter model training.
    """
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None

    tensors = torch.stack([item[0] for item in valid_items], dim=0)  # [B, 16, 3, 224, 224]
    labels = torch.stack([item[1] for item in valid_items], dim=0)   # [B, 1]
    metas = [item[2] for item in valid_items]

    return tensors, labels, metas
