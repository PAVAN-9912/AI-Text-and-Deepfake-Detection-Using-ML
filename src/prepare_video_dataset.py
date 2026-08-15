from pathlib import Path
import shutil
import random
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path(
    r"datasets\video\celebdf_raw"
)

OUTPUT_DIR = Path(
    r"datasets\video\prepared"
)

TEST_LIST = RAW_DIR / "List_of_testing_videos.txt"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
VALIDATION_RATIO = 0.20


# ============================================================
# HELPER
# ============================================================

def normalize_path(path_string):
    """
    Convert a dataset-relative path into a consistent
    lowercase POSIX-style representation.
    """

    return (
        str(path_string)
        .replace("\\", "/")
        .strip()
        .lower()
    )


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# CHECK DATASET
# ============================================================

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {RAW_DIR}"
    )


if not TEST_LIST.exists():
    raise FileNotFoundError(
        f"Testing list not found: {TEST_LIST}"
    )


# ============================================================
# READ OFFICIAL TEST LIST
# ============================================================

test_entries = []

with open(
    TEST_LIST,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            continue

        label = int(parts[0])

        relative_path = normalize_path(
            parts[1]
        )

        test_entries.append(
            {
                "label": label,
                "relative_path": relative_path
            }
        )


print(
    "Official testing videos:",
    len(test_entries)
)


# ============================================================
# OFFICIAL TEST PATHS
# ============================================================

test_paths = {
    item["relative_path"]
    for item in test_entries
}


# ============================================================
# DISCOVER ALL VIDEOS
# ============================================================

real_dirs = [
    RAW_DIR / "Celeb-real",
    RAW_DIR / "YouTube-real"
]

fake_dir = RAW_DIR / "Celeb-synthesis"


all_videos = []


# ============================================================
# DISCOVER REAL VIDEOS
# ============================================================

for directory in real_dirs:

    if not directory.exists():

        raise FileNotFoundError(
            f"Missing directory: {directory}"
        )

    for video in directory.glob("*.mp4"):

        relative_path = normalize_path(
            video.relative_to(
                RAW_DIR
            ).as_posix()
        )

        all_videos.append(
            {
                "source_path": video,
                "relative_path": relative_path,
                "label": 1
            }
        )


# ============================================================
# DISCOVER FAKE VIDEOS
# ============================================================

if not fake_dir.exists():

    raise FileNotFoundError(
        f"Missing directory: {fake_dir}"
    )


for video in fake_dir.glob("*.mp4"):

    relative_path = normalize_path(
        video.relative_to(
            RAW_DIR
        ).as_posix()
    )

    all_videos.append(
        {
            "source_path": video,
            "relative_path": relative_path,
            "label": 0
        }
    )


print(
    "Total discovered videos:",
    len(all_videos)
)


# ============================================================
# CHECK TEST MATCHING
# ============================================================

discovered_paths = {
    item["relative_path"]
    for item in all_videos
}


matched_test_paths = (
    test_paths
    & discovered_paths
)


missing_test_paths = (
    test_paths
    - discovered_paths
)


print(
    "Matched official test paths:",
    len(matched_test_paths)
)


print(
    "Missing official test paths:",
    len(missing_test_paths)
)


if len(matched_test_paths) != len(test_paths):

    print(
        "\nWARNING: Some official test videos "
        "were not found."
    )

    for path in sorted(
        missing_test_paths
    )[:20]:

        print(
            "Missing:",
            path
        )

    raise RuntimeError(
        "Official test-set matching failed. "
        "Dataset preparation stopped."
    )


# ============================================================
# SEPARATE TEST AND DEVELOPMENT DATA
# ============================================================

test_videos = []
development_videos = []


for item in all_videos:

    if item["relative_path"] in test_paths:

        test_videos.append(item)

    else:

        development_videos.append(item)


print(
    "\nTest videos:",
    len(test_videos)
)


print(
    "Development videos:",
    len(development_videos)
)


# ============================================================
# TEST LABEL COUNTS
# ============================================================

test_real = sum(
    item["label"] == 1
    for item in test_videos
)


test_fake = sum(
    item["label"] == 0
    for item in test_videos
)


print(
    "Test real:",
    test_real
)


print(
    "Test fake:",
    test_fake
)


# ============================================================
# DEVELOPMENT DATA BY CLASS
# ============================================================

development_real = [
    item
    for item in development_videos
    if item["label"] == 1
]


development_fake = [
    item
    for item in development_videos
    if item["label"] == 0
]


random.shuffle(
    development_real
)

random.shuffle(
    development_fake
)


# ============================================================
# VALIDATION COUNTS
# ============================================================

real_val_count = int(
    len(development_real)
    * VALIDATION_RATIO
)


fake_val_count = int(
    len(development_fake)
    * VALIDATION_RATIO
)


# ============================================================
# VALIDATION
# ============================================================

validation_videos = (
    development_real[
        :real_val_count
    ]
    +
    development_fake[
        :fake_val_count
    ]
)


# ============================================================
# TRAINING
# ============================================================

train_videos = (
    development_real[
        real_val_count:
    ]
    +
    development_fake[
        fake_val_count:
    ]
)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    train_videos
)

random.shuffle(
    validation_videos
)

random.shuffle(
    test_videos
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\nDATASET SPLIT")
print("=============")


print(
    "Training:",
    len(train_videos)
)


print(
    "Validation:",
    len(validation_videos)
)


print(
    "Test:",
    len(test_videos)
)


print("\nTRAIN LABELS")


print(
    "Real:",
    sum(
        item["label"] == 1
        for item in train_videos
    )
)


print(
    "Fake:",
    sum(
        item["label"] == 0
        for item in train_videos
    )
)


print("\nVALIDATION LABELS")


print(
    "Real:",
    sum(
        item["label"] == 1
        for item in validation_videos
    )
)


print(
    "Fake:",
    sum(
        item["label"] == 0
        for item in validation_videos
    )
)


print("\nTEST LABELS")


print(
    "Real:",
    sum(
        item["label"] == 1
        for item in test_videos
    )
)


print(
    "Fake:",
    sum(
        item["label"] == 0
        for item in test_videos
    )
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in [
    "train",
    "validation",
    "test"
]:

    for label_name in [
        "real",
        "fake"
    ]:

        directory = (
            OUTPUT_DIR
            / split
            / label_name
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_videos(
    videos,
    split
):

    records = []

    total = len(videos)

    for index, item in enumerate(
        videos,
        start=1
    ):

        label_name = (
            "real"
            if item["label"] == 1
            else "fake"
        )


        destination_dir = (
            OUTPUT_DIR
            / split
            / label_name
        )


        destination = (
            destination_dir
            / item["source_path"].name
        )


        shutil.copy2(
            item["source_path"],
            destination
        )


        records.append(
            {
                "split": split,
                "label": item["label"],
                "label_name": label_name,
                "filename": (
                    item["source_path"].name
                ),
                "relative_path": (
                    item["relative_path"]
                ),
                "prepared_path": str(
                    destination
                ),
                "source": (
                    "official_test"
                    if split == "test"
                    else "development"
                )
            }
        )


        if (
            index % 100 == 0
            or index == total
        ):

            print(
                f"{split}: "
                f"copied {index}/{total}"
            )


    return records


# ============================================================
# COPY TRAINING DATA
# ============================================================

print(
    "\nCopying training videos..."
)


train_records = copy_videos(
    train_videos,
    "train"
)


# ============================================================
# COPY VALIDATION DATA
# ============================================================

print(
    "\nCopying validation videos..."
)


validation_records = copy_videos(
    validation_videos,
    "validation"
)


# ============================================================
# COPY OFFICIAL TEST DATA
# ============================================================

print(
    "\nCopying official test videos..."
)


test_records = copy_videos(
    test_videos,
    "test"
)


# ============================================================
# CREATE METADATA
# ============================================================

metadata = pd.DataFrame(
    train_records
    +
    validation_records
    +
    test_records
)


metadata_path = (
    OUTPUT_DIR
    / "metadata.csv"
)


metadata.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print(
    "\nPREPARATION COMPLETED"
)

print(
    "====================="
)


print(
    "Total prepared videos:",
    len(metadata)
)


print(
    "Metadata:",
    metadata_path
)


print(
    "\nFINAL DISTRIBUTION"
)


print(
    metadata.groupby(
        [
            "split",
            "label_name"
        ]
    ).size()
)


# ============================================================
# FINAL TEST ASSERTION
# ============================================================

assert (
    len(test_videos) == 518
), (
    "ERROR: Official test set "
    "must contain exactly 518 videos."
)


print(
    "\nOfficial 518-video test set: VERIFIED"
)