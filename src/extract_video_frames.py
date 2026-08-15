from pathlib import Path
import cv2
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PREPARED_DIR = Path(
    r"datasets\video\prepared"
)

FRAMES_DIR = Path(
    r"datasets\video\frames"
)

METADATA_PATH = (
    PREPARED_DIR / "metadata.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FRAMES_PER_VIDEO = 8

FRAME_WIDTH = 224
FRAME_HEIGHT = 224

JPEG_QUALITY = 90


# ============================================================
# CHECK PATHS
# ============================================================

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}"
    )


FRAMES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)


print(
    "VIDEO FRAME EXTRACTION"
)

print(
    "======================"
)

print(
    "Total videos:",
    len(metadata)
)

print(
    "Frames per video:",
    FRAMES_PER_VIDEO
)


# ============================================================
# EXTRACTION FUNCTION
# ============================================================

def extract_frames(
    video_path,
    output_dir
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return 0


    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if frame_count <= 0:

        cap.release()

        return 0


    if fps <= 0:

        fps = 25.0


    # --------------------------------------------------------
    # Select evenly spaced frame indices
    # --------------------------------------------------------

    if frame_count <= FRAMES_PER_VIDEO:

        indices = np.arange(
            frame_count
        )

    else:

        indices = np.linspace(
            0,
            frame_count - 1,
            FRAMES_PER_VIDEO,
            dtype=int
        )


    saved = 0


    for frame_index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )


        success, frame = cap.read()


        if not success:

            continue


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        frame_name = (
            f"frame_{int(frame_index):06d}.jpg"
        )


        output_path = (
            output_dir / frame_name
        )


        cv2.imwrite(
            str(output_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )


        saved += 1


    cap.release()


    return saved


# ============================================================
# PROCESS VIDEOS
# ============================================================

records = []

failed = []

total = len(metadata)


for index, row in metadata.iterrows():

    video_path = Path(
        row["prepared_path"]
    )


    # --------------------------------------------------------
    # Folder structure
    # --------------------------------------------------------

    split = row["split"]

    label_name = row["label_name"]

    video_id = (
        Path(
            row["filename"]
        ).stem
    )


    output_dir = (
        FRAMES_DIR
        / split
        / label_name
        / video_id
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Skip if already extracted
    # --------------------------------------------------------

    existing_frames = list(
        output_dir.glob("*.jpg")
    )


    if len(existing_frames) >= FRAMES_PER_VIDEO:

        saved = len(existing_frames)

    else:

        saved = extract_frames(
            video_path,
            output_dir
        )


    if saved == 0:

        failed.append(
            {
                "split": split,
                "label": row["label"],
                "filename": row["filename"],
                "path": str(video_path)
            }
        )


    records.append(
        {
            "split": split,
            "label": row["label"],
            "label_name": label_name,
            "video_id": video_id,
            "filename": row["filename"],
            "frame_count": saved,
            "frame_directory": str(
                output_dir
            )
        }
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    current = index + 1


    if (
        current % 50 == 0
        or current == total
    ):

        print(
            f"Processed "
            f"{current}/{total}"
        )


# ============================================================
# SAVE FRAME METADATA
# ============================================================

frame_metadata = pd.DataFrame(
    records
)


frame_metadata_path = (
    FRAMES_DIR
    / "frame_metadata.csv"
)


frame_metadata.to_csv(
    frame_metadata_path,
    index=False
)


# ============================================================
# SAVE FAILURES
# ============================================================

failed_path = (
    FRAMES_DIR
    / "failed_videos.csv"
)


pd.DataFrame(
    failed
).to_csv(
    failed_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nFRAME EXTRACTION COMPLETED"
)

print(
    "==========================="
)

print(
    "Videos processed:",
    len(records)
)

print(
    "Videos with extraction failure:",
    len(failed)
)

print(
    "Total frames:",
    frame_metadata[
        "frame_count"
    ].sum()
)

print(
    "Frame metadata:",
    frame_metadata_path
)

print(
    "Failed videos:",
    failed_path
)


print(
    "\nFRAMES BY SPLIT"
)

print(
    frame_metadata.groupby(
        ["split", "label_name"]
    )["frame_count"]
    .agg(
        ["count", "sum"]
    )
)