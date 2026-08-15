import os
import random
import numpy as np
import pandas as pd

from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

FRAME_METADATA_PATH = Path(
    r"datasets\video\frames\frame_metadata.csv"
)

OUTPUT_PATH = Path(
    r"datasets\video\temporal_resnet_features.csv"
)

CHECKPOINT_PATH = Path(
    r"datasets\video\temporal_resnet_checkpoint.csv"
)

SEQUENCE_LENGTH = 8

FEATURE_DIMENSION = 512

BATCH_SIZE = 32

CHECKPOINT_EVERY = 25

RANDOM_STATE = 42


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)


# ============================================================
# HEADER
# ============================================================

print(
    "TEMPORAL RESNET-18 FEATURE EXTRACTION"
)

print(
    "======================================"
)

print(
    "Device:",
    DEVICE
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Feature dimension:",
    FEATURE_DIMENSION
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    FRAME_METADATA_PATH
)


print(
    "\nTotal videos:",
    len(metadata)
)


# ============================================================
# VERIFY METADATA
# ============================================================

required_columns = [
    "split",
    "label",
    "label_name",
    "video_id",
    "filename",
    "frame_count",
    "frame_directory"
]


missing_columns = [
    column
    for column in required_columns
    if column not in metadata.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns: "
        +
        str(missing_columns)
    )


# ============================================================
# LOAD RESNET-18
# ============================================================

print(
    "\nLoading ResNet-18..."
)


weights = (
    ResNet18_Weights.DEFAULT
)


resnet = resnet18(
    weights=weights
)


# Remove final classification layer.
#
# Original:
#
# 512 features → 1000 ImageNet classes
#
# We need:
#
# 512-dimensional visual representation
#

resnet.fc = nn.Identity()


resnet = resnet.to(
    DEVICE
)


resnet.eval()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = (
    weights.transforms()
)


# ============================================================
# CHECKPOINT SUPPORT
# ============================================================

processed_ids = set()

results = []


if CHECKPOINT_PATH.exists():

    print(
        "\nCheckpoint found:"
    )

    print(
        CHECKPOINT_PATH
    )


    checkpoint_df = pd.read_csv(
        CHECKPOINT_PATH
    )


    if "video_id" in checkpoint_df.columns:

        processed_ids = set(
            checkpoint_df[
                "video_id"
            ]
            .astype(str)
        )


        results = (
            checkpoint_df
            .to_dict(
                "records"
            )
        )


        print(
            "Videos already processed:",
            len(processed_ids)
        )


else:

    print(
        "\nStarting feature extraction from scratch."
    )


# ============================================================
# FRAME PATH FUNCTION
# ============================================================

def get_frame_paths(
    frame_directory
):

    directory = Path(
        frame_directory
    )


    if not directory.exists():

        return []


    frame_paths = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ]
    )


    return frame_paths


# ============================================================
# PAD / SAMPLE FRAMES
# ============================================================

def prepare_frames(
    frame_paths
):

    if len(frame_paths) == 0:

        return []


    # --------------------------------------------------------
    # Normal case:
    # exactly 8 frames
    # --------------------------------------------------------

    if len(frame_paths) == SEQUENCE_LENGTH:

        return frame_paths


    # --------------------------------------------------------
    # More than 8:
    # uniformly sample 8 frames
    # --------------------------------------------------------

    if len(frame_paths) > SEQUENCE_LENGTH:

        indices = np.linspace(
            0,
            len(frame_paths) - 1,
            SEQUENCE_LENGTH
        ).astype(int)


        return [
            frame_paths[index]
            for index in indices
        ]


    # --------------------------------------------------------
    # Fewer than 8:
    #
    # Repeat the final available frame.
    #
    # The dataset currently has exactly one such video:
    #
    # id27_0005
    #
    # with only one extracted frame.
    # --------------------------------------------------------

    padded = list(
        frame_paths
    )


    while len(padded) < SEQUENCE_LENGTH:

        padded.append(
            frame_paths[-1]
        )


    return padded


# ============================================================
# EXTRACT FEATURES FOR ONE VIDEO
# ============================================================

def extract_video_features(
    frame_directory
):

    frame_paths = get_frame_paths(
        frame_directory
    )


    prepared_frames = prepare_frames(
        frame_paths
    )


    if len(prepared_frames) != SEQUENCE_LENGTH:

        return None


    images = []


    for frame_path in prepared_frames:

        try:

            image = Image.open(
                frame_path
            ).convert(
                "RGB"
            )


            image = transform(
                image
            )


            images.append(
                image
            )


        except Exception as error:

            print(
                "\nFrame error:",
                frame_path
            )

            print(
                error
            )

            return None


    # --------------------------------------------------------
    # Batch all 8 frames
    # --------------------------------------------------------

    batch = torch.stack(
        images
    ).to(
        DEVICE
    )


    with torch.no_grad():

        features = resnet(
            batch
        )


    features = (
        features
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    # Expected:
    #
    # (8, 512)
    #

    if features.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    ):

        raise ValueError(
            "Unexpected feature shape: "
            +
            str(features.shape)
        )


    return features


# ============================================================
# FEATURE COLUMN NAMES
# ============================================================

feature_columns = []

for frame_number in range(
    1,
    SEQUENCE_LENGTH + 1
):

    for feature_number in range(
        FEATURE_DIMENSION
    ):

        feature_columns.append(
            f"frame_{frame_number}_feature_{feature_number}"
        )


# ============================================================
# PROCESS VIDEOS
# ============================================================

total_videos = len(
    metadata
)

processed_this_run = 0

failed_videos = []


for index, row in metadata.iterrows():

    video_id = str(
        row["video_id"]
    )


    # --------------------------------------------------------
    # Skip checkpointed videos
    # --------------------------------------------------------

    if video_id in processed_ids:

        continue


    frame_directory = row[
        "frame_directory"
    ]


    features = extract_video_features(
        frame_directory
    )


    if features is None:

        failed_videos.append(
            video_id
        )

        print(
            f"\nFAILED: {video_id}"
        )

        continue


    # --------------------------------------------------------
    # Flatten:
    #
    # 8 × 512
    #
    # into:
    #
    # 4096 values
    # --------------------------------------------------------

    flattened = (
        features
        .reshape(-1)
        .tolist()
    )


    record = {

        "split":
            row["split"],

        "label":
            int(row["label"]),

        "label_name":
            row["label_name"],

        "video_id":
            video_id,

        "filename":
            row["filename"],

        "frame_count":
            int(row["frame_count"]),

        "frame_directory":
            row["frame_directory"]

    }


    for column, value in zip(
        feature_columns,
        flattened
    ):

        record[column] = value


    results.append(
        record
    )


    processed_ids.add(
        video_id
    )


    processed_this_run += 1


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if (
        processed_this_run
        %
        CHECKPOINT_EVERY
        == 0
    ):

        checkpoint_df = pd.DataFrame(
            results
        )


        checkpoint_df.to_csv(
            CHECKPOINT_PATH,
            index=False
        )


        print(
            f"Checkpoint saved: "
            f"{len(processed_ids)}/{total_videos}"
        )


    elif (
        processed_this_run % 10 == 0
        or
        len(processed_ids) == total_videos
    ):

        print(
            f"Processed: "
            f"{len(processed_ids)}/{total_videos}"
        )


# ============================================================
# SAVE FINAL DATASET
# ============================================================

print(
    "\nFINALIZING FEATURE DATASET..."
)


result_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Remove accidental duplicates
# ------------------------------------------------------------

result_df = (
    result_df
    .drop_duplicates(
        subset=["video_id"]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# Sort according to original metadata order
# ------------------------------------------------------------

metadata_order = {
    str(video_id): index
    for index, video_id
    in enumerate(
        metadata["video_id"]
    )
}


result_df["_order"] = (
    result_df["video_id"]
    .map(
        metadata_order
    )
)


result_df = (
    result_df
    .sort_values(
        "_order"
    )
    .drop(
        columns=["_order"]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\nFEATURE EXTRACTION COMPLETED"
)

print(
    "============================="
)

print(
    "Videos with features:",
    len(result_df)
)

print(
    "Feature dimensions per frame:",
    FEATURE_DIMENSION
)

print(
    "Frames per video:",
    SEQUENCE_LENGTH
)

print(
    "Total temporal features:",
    SEQUENCE_LENGTH
    *
    FEATURE_DIMENSION
)

print(
    "Output:",
    OUTPUT_PATH
)


print(
    "\nFEATURE MATRIX SHAPE"
)

print(
    (
        len(result_df),
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION
    )
)


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print(
    "\nSPLIT DISTRIBUTION"
)

print(
    result_df.groupby(
        [
            "split",
            "label_name"
        ]
    ).size()
)


# ============================================================
# FRAME COUNT DISTRIBUTION
# ============================================================

print(
    "\nORIGINAL FRAME COUNT DISTRIBUTION"
)

print(
    result_df[
        "frame_count"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# FAILED VIDEOS
# ============================================================

if failed_videos:

    failed_path = (
        OUTPUT_PATH.parent
        /
        "temporal_failed_videos.csv"
    )


    pd.DataFrame(
        {
            "video_id":
                failed_videos
        }
    ).to_csv(
        failed_path,
        index=False
    )


    print(
        "\nFailed videos:",
        len(failed_videos)
    )

    print(
        "Failed video list:",
        failed_path
    )

else:

    print(
        "\nFailed videos: 0"
    )


# ============================================================
# CHECKPOINT CLEANUP
# ============================================================

if CHECKPOINT_PATH.exists():

    try:

        CHECKPOINT_PATH.unlink()

        print(
            "Checkpoint removed after successful completion."
        )

    except Exception:

        print(
            "Warning: could not remove checkpoint."
        )


print(
    "\nTEMPORAL RESNET FEATURE EXTRACTION COMPLETED"
)