from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# PATHS
# ============================================================

FRAMES_DIR = Path(
    r"datasets\video\frames"
)

METADATA_PATH = (
    FRAMES_DIR / "frame_metadata.csv"
)

OUTPUT_PATH = Path(
    r"datasets\video\resnet_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD METADATA
# ============================================================

if not METADATA_PATH.exists():

    raise FileNotFoundError(
        f"Frame metadata not found: {METADATA_PATH}"
    )


metadata = pd.read_csv(
    METADATA_PATH
)


print(
    "RESNET-18 VIDEO FEATURE EXTRACTION"
)

print(
    "=================================="
)

print(
    "Videos:",
    len(metadata)
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD PRETRAINED RESNET-18
# ============================================================

print(
    "\nLoading ResNet-18..."
)


weights = (
    ResNet18_Weights.DEFAULT
)


model = resnet18(
    weights=weights
)


# Remove final classification layer.
# Output becomes the 512-dimensional
# feature representation.

model.fc = torch.nn.Identity()


model = model.to(
    DEVICE
)


model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = weights.transforms()


# ============================================================
# DATASET
# ============================================================

class FrameDataset(Dataset):

    def __init__(
        self,
        frame_paths
    ):

        self.frame_paths = frame_paths


    def __len__(self):

        return len(
            self.frame_paths
        )


    def __getitem__(
        self,
        index
    ):

        path = self.frame_paths[
            index
        ]


        try:

            image = Image.open(
                path
            ).convert(
                "RGB"
            )


            image = transform(
                image
            )


            return image, str(path)


        except Exception as e:

            print(
                f"Image error: {path}"
            )

            print(e)


            return (
                torch.zeros(
                    3,
                    224,
                    224
                ),
                str(path)
            )


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

video_features = []


total_videos = len(
    metadata
)


for video_index, row in metadata.iterrows():

    frame_directory = Path(
        row["frame_directory"]
    )


    frame_paths = sorted(
        frame_directory.glob(
            "*.jpg"
        )
    )


    if not frame_paths:

        print(
            "WARNING: No frames:",
            row["filename"]
        )

        continue


    dataset = FrameDataset(
        frame_paths
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    frame_features = []


    with torch.no_grad():

        for images, paths in loader:

            images = images.to(
                DEVICE
            )


            features = model(
                images
            )


            features = (
                features
                .cpu()
                .numpy()
            )


            frame_features.append(
                features
            )


    if not frame_features:

        continue


    frame_features = np.vstack(
        frame_features
    )


    # --------------------------------------------------------
    # Average frame features
    # --------------------------------------------------------

    video_feature = (
        frame_features.mean(
            axis=0
        )
    )


    # --------------------------------------------------------
    # Build record
    # --------------------------------------------------------

    record = {

        "video_id":
            Path(
                row["filename"]
            ).stem,

        "filename":
            row["filename"],

        "split":
            row["split"],

        "label":
            row["label"],

        "label_name":
            row["label_name"],

        "num_frames":
            len(frame_paths)
    }


    # --------------------------------------------------------
    # Add 512 feature columns
    # --------------------------------------------------------

    for feature_index, value in enumerate(
        video_feature
    ):

        record[
            f"feature_{feature_index}"
        ] = float(value)


    video_features.append(
        record
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    current = video_index + 1


    if (
        current % 25 == 0
        or current == total_videos
    ):

        print(
            f"Processed "
            f"{current}/{total_videos}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    video_features
)


# ============================================================
# SAVE
# ============================================================

features_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nFEATURE EXTRACTION COMPLETED"
)

print(
    "============================="
)

print(
    "Videos with features:",
    len(features_df)
)

print(
    "Feature dimensions:",
    512
)

print(
    "Output:",
    OUTPUT_PATH
)


print(
    "\nSPLIT DISTRIBUTION"
)

print(
    features_df.groupby(
        [
            "split",
            "label_name"
        ]
    ).size()
)


print(
    "\nFEATURE MATRIX SHAPE"
)

print(
    features_df[
        [
            f"feature_{i}"
            for i in range(512)
        ]
    ].shape
)